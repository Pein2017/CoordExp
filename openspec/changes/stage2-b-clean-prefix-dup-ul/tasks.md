## 1. OpenSpec Contract Hardening

- [x] 1.1 Add delta spec files for:
  - `stage2-ab-training`
  - `teacher-forcing-objective-pipeline`
  - `teacher-forcing-unified-loss-registry`
  - `trainer-metrics-components`
- [x] 1.2 Rewrite the proposal/design around a single canonical Channel-B contract:
  - no raw-prefix compatibility mode,
  - no legacy knob aliases,
  - no backward-compatibility promises.
- [x] 1.3 Make the parsing contract explicit as bounded container salvage plus strict record acceptance.

## 2. Typed Config and Pipeline Structure

- [x] 2.1 Define the strict `stage2_ab.channel_b` surface as:
  - `duplicate_iou_threshold`
  - `producer_wait_timeout_s`
  - `ddp_phase_timeout_s`
- [x] 2.2 Add `duplicate_ul` as a new explicit objective module in the canonical Stage-2 AB pipeline ordering:
  - `token_ce`
  - `duplicate_ul`
  - `bbox_geo`
  - `coord_reg`
- [x] 2.3 Keep `duplicate_ul` simple in v1:
  - `channels: [B]`
  - module `weight` is the only scaling surface
  - `config: {}`
- [x] 2.4 Remove the old raw-prefix and invalid-structure-multiplier compatibility story from the spec; old placements should fail fast instead of being preserved.

## 3. Channel-B Clean-prefix Data Products

- [x] 3.1 Define the ordered per-sample intermediates explicitly:
  - `parsed_bbox_objects_raw`
  - `accepted_objects_clean`
  - `duplicate_bursts_by_boundary`
  - `match_on_clean`
  - `clean_teacher_forced_prefix`
  - `clean_teacher_forced_target`
  - `duplicate_ul_targets`
- [x] 3.2 Define sequential dedup as bbox-only in v1:
  - shared `normalize_desc`
  - exact normalized-desc equality
  - `IoU >= duplicate_iou_threshold`
  - compare only against previously accepted clean bbox objects
- [x] 3.3 Define clean boundary indexing as insertion-position based and include the pre-first / empty-clean-prefix case explicitly.

## 4. Matching and Teacher-forced Target Construction

- [x] 4.1 Make matching and FN detection run on `accepted_objects_clean`, not on the raw parsed bbox list.
- [x] 4.2 Define the positive Channel-B teacher-forced prefix as canonical serialization of `accepted_objects_clean`, not raw rollout-prefix token ids.
- [x] 4.3 Keep generic unmatched clean extras in the clean prefix as neutral context only:
  - they must not populate matched-prefix struct masks,
  - they must not populate coord/bbox supervision groups,
  - they must not become duplicate-ul positives.
- [x] 4.4 Keep closure/EOS supervision and canonical FN append behavior unchanged in intent, but relative to the clean teacher-forced target.

## 5. Duplicate UL Objective Module

- [x] 5.1 Add the `duplicate_ul` objective module contract across:
  - objective pipeline module registry,
  - typed pipeline allowlists,
  - runtime module dispatch,
  - trainer-side objective logging.
- [x] 5.2 Define duplicate-UL token provenance canonically:
  - `clean_continuation(boundary)` comes from the clean teacher-forced target,
  - `duplicate_continuation(boundary, dup)` comes from canonical serialization of that duplicate object at the same boundary.
- [x] 5.3 Define the UL target as the first true LCP-divergence token.
- [x] 5.4 Aggregate duplicate UL as one unit term per unique divergence token per clean boundary.
- [x] 5.5 Skip continuations with no safe divergence token and count them explicitly in diagnostics.

## 6. Metrics and Diagnostics

- [x] 6.1 Add the new objective atom key:
  - `loss/B_rollout_text/duplicate_ul`
- [x] 6.2 Add duplicate-collapse gauges:
  - `dup/max_desc_count`
  - `dup/saturation_rate`
- [x] 6.3 Add aggregation-safe duplicate counters:
  - `dup/near_iou90_pairs_same_desc_count`
  - `dup/near_iou90_pairs_any_desc_count`
  - `stage2_ab/channel_b/dup/N_raw_bbox_valid`
  - `stage2_ab/channel_b/dup/N_clean_accepted`
  - `stage2_ab/channel_b/dup/N_duplicates`
  - `stage2_ab/channel_b/dup/N_duplicate_bursts`
  - `stage2_ab/channel_b/dup/N_ul_boundaries`
  - `stage2_ab/channel_b/dup/N_ul_skipped_no_divergence`
- [x] 6.4 Review eval-side duplicate diagnostics naming/artifacts if they land in the same implementation.
  No eval-side duplicate diagnostics were added in this implementation, so no eval artifact rename was required.

## 7. Validation Plan

- [x] 7.1 Add config-contract coverage for:
  - `stage2_ab.channel_b.duplicate_iou_threshold`
  - `duplicate_ul` module acceptance
  - rejection of old raw-prefix / invalid-multiplier placements
- [x] 7.2 Add lightweight training-unit coverage for:
  - bounded container salvage vs strict record acceptance behavior,
  - dedup-before-Hungarian clean matching,
  - clean-prefix teacher forcing for later true positives,
  - boundary indexing including pre-first / empty-clean-prefix cases,
  - first true LCP-divergence token identification,
  - same-boundary duplicate bursts that share a divergence token collapsing to one UL term,
  - unsafe or unavailable divergence token causing duplicate-ul skip plus counter increment,
  - neutral unmatched clean extras producing no CE / geo / coord supervision,
  - duplicate-ul logging and counter aggregation behavior.
- [x] 7.3 Add a small Stage-2 config materialization or smoke path that exercises the canonical clean-prefix pipeline shape.

Validation commands expected after implementation:

- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_teacher_forcing_token_ce.py`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_pending_metrics_aggregation.py`
- `PYTHONPATH=. conda run -n ms python -c "from src.config.loader import ConfigLoader; ConfigLoader.load_materialized_training_config('configs/stage2_two_channel/base.yaml'); print('OK')"`

## 8. Docs Sync After Implementation

- [x] 8.1 Update `docs/training/STAGE2_RUNBOOK.md` to describe the canonical clean-prefix Channel-B contract and remove the raw-prefix / immutable-prefix wording.
- [x] 8.2 Update `docs/training/METRICS.md` with:
  - `loss/B_rollout_text/duplicate_ul`
  - duplicate-collapse gauges and counters
  - any changed Channel-B target-construction wording
- [x] 8.3 Update `docs/ARTIFACTS.md` if the implementation changes documented run artifacts, metric families, or reproducibility surfaces.
- [x] 8.4 Review `docs/eval/README.md` if eval-side duplicate diagnostics or duplicate-analysis artifacts land as part of this feature.
  Reviewed: no eval-side duplicate diagnostics or duplicate-analysis artifacts landed, so no doc change was required.
- [x] 8.5 Update `docs/README.md` if the docs index or recommended reading order needs to surface any newly added/renamed training guidance.
- [x] 8.6 Sync the landed clean-prefix contract into main `openspec/specs/*` and refresh the `coordexp-codebase` skill guidance so repo navigation points to the v2 Channel-B surfaces.

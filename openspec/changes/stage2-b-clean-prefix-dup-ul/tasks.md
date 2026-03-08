## 1. Infrastructure Audit + Contract Wiring

- [ ] 1.1 Document the current Stage-2 Channel-B data flow in code comments and change notes:
  - rollout generation,
  - strict parse + bbox-valid filtering,
  - matching,
  - raw-prefix teacher-forced target construction,
  - Channel-A / Channel-B branching,
  - objective-pipeline execution and loss assembly.
- [ ] 1.2 Extend the Stage-2 typed config/runtime wiring for the new v2 knobs:
  - sequential dedup enable/disable,
  - duplicate IoU threshold,
  - duplicate UL weight,
  - any safety cap needed for boundary token targets,
  - explicit objective atom enablement for duplicate UL.
- [ ] 1.3 Keep old configs usable where possible, but ensure the new v2 profile is explicit and auditable.

## 2. Channel-B Clean-prefix Data Products

- [ ] 2.1 Add a deterministic sequential dedup helper over bbox-valid parsed rollout objects:
  - bbox-only in v1,
  - shared `normalize_desc`,
  - configurable `tau_dup`,
  - compare each candidate only against previously accepted clean bbox objects.
- [ ] 2.2 Produce Channel-B clean data products:
  - `accepted_objects_clean`,
  - duplicate objects,
  - duplicate bursts attached to clean boundaries,
  - per-boundary metadata needed to reconstruct duplicate continuations.
- [ ] 2.3 Preserve existing strict parse / drop behavior for invalid or truncated rollouts.

## 3. Matching + Clean Teacher-forced Target Construction

- [ ] 3.1 Update Channel-B matching and FN detection to run on the clean accepted bbox sequence.
- [ ] 3.2 Rebuild the Channel-B assistant target from the clean accepted sequence plus existing FN injection rules.
- [ ] 3.3 Ensure later correct objects are teacher-forced on the clean accepted prefix, not the raw duplicate-contaminated prefix.
- [ ] 3.4 Preserve current FP-neutral semantics for generic unmatched clean accepted objects and existing closure/EOS supervision.

## 4. Duplicate UL Objective Atom

- [ ] 4.1 Add a new explicit teacher-forcing objective atom/module for duplicate UL rather than extending `token_ce`.
- [ ] 4.2 At each clean boundary with a duplicate burst:
  - derive the clean continuation from the clean teacher-forced target,
  - derive duplicate continuations from the duplicate burst,
  - compute the first true LCP-divergence token relative to the clean continuation,
  - deduplicate divergence tokens per boundary,
  - apply one unit-weight UL term per unique divergence token.
- [ ] 4.3 Implement safe skip behavior when no safe divergence token exists and surface that in metrics.
- [ ] 4.4 Keep generic unmatched clean objects neutral; do not add negative CE or whole-span suppression for them.

## 5. Metrics, Monitoring, and Debug Surfaces

- [ ] 5.1 Add duplicate-collapse diagnostics for training/eval:
  - `dup/max_desc_count`
  - `dup/near_iou90_pairs_same_desc`
  - `dup/near_iou90_pairs_any_desc`
  - `dup/saturation_rate`
- [ ] 5.2 Add Channel-B v2 counters:
  - raw parsed objects,
  - clean accepted objects,
  - duplicate objects,
  - duplicate bursts,
  - boundaries with UL applied,
  - duplicate continuations skipped for no safe divergence token.
- [ ] 5.3 Extend any existing monitor dump or debug artifact path minimally so duplicate analysis is inspectable.

## 6. Configs and Recommended Profiles

- [ ] 6.1 Add the new Stage-2 v2 config knobs to the canonical typed YAML surface.
- [ ] 6.2 Update or add a recommended Stage-2 v2 profile that is safer than the current B-hot long-rollout setup:
  - A-hot / B-cold by default,
  - less permissive `max_new_tokens` where appropriate,
  - clear naming that distinguishes the new clean-prefix duplicate-UL contract.
- [ ] 6.3 Keep unrelated defaults unchanged unless required by the new contract.

## 7. Validation

- [ ] 7.1 Add lightweight tests for:
  - sequential dedup on synthetic bbox objects,
  - duplicate burst extraction,
  - LCP-divergence token identification,
  - clean-prefix target construction,
  - loss-mask behavior showing:
    - later correct objects use the clean prefix,
    - duplicate-certified continuations contribute UL,
    - generic unmatched clean objects remain negative-neutral.
- [ ] 7.2 Extend config-contract tests for the new typed knobs and explicit objective atom.
- [ ] 7.3 Add at least one small Stage-2 sanity path or smoke config that exercises the v2 Channel-B contract without broad runtime changes.

Validation commands (expected after implementation):

- `conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- `conda run -n ms python -m pytest -q tests/test_teacher_forcing_token_ce.py`
- `conda run -n ms python -m pytest -q tests/test_stage2_pending_metrics_aggregation.py`

## 1. Scaffolding (No behavior change)
- [ ] 1.1 Add `src/trainers/batch_extras.py` (stable key constants + `BatchExtras` + pop helper)
- [ ] 1.2 Add `src/trainers/metrics/reporter.py` (ms-swift `custom_metrics` adapter; warn-once helper)

## 2. Collator Enrichment (Composable, Back-Compat)
- [ ] 2.1 Add `src/data_collators/enrichers.py` with small enrichers:
      - dataset meta (labels/segments/pack_num_samples)
      - token types
      - instability meta
- [ ] 2.2 Refactor `src/data_collators/dataset_metrics.py` to orchestrate enrichers (same emitted keys)

## 3. Metric Computation (Aggregate-Only)
- [ ] 3.1 Add `src/trainers/metrics/aggregate_token_metrics.py` (pure computations)
- [ ] 3.2 Add `src/trainers/metrics/coord_monitors.py` (coord flip + mass diagnostics)
- [ ] 3.3 Refactor `AggregateTokenTypeMetricsMixin` to call pure functions + reporter

## 4. Loss Components (Fail-Fast)
- [ ] 4.1 Add `src/trainers/losses/coord_soft_ce_w1.py` and move core logic behind `CoordSoftCEW1LossMixin`
- [ ] 4.2 Keep fail-fast behavior for enabled coord loss; preserve label/acc semantics

## 5. Monitoring (Best-Effort)
- [ ] 5.1 Add `src/trainers/monitoring/instability.py` and move I/O/guard logic behind `InstabilityMonitorMixin`

## 6. Tests + Documentation
- [ ] 6.1 Add unit tests for pack metadata + token-type alignment (packed and non-packed)
- [ ] 6.2 Add unit tests for coord-loss masking + loss composition on toy logits
- [ ] 6.3 Update any internal docs if new module layout affects contributor workflow
- [ ] 6.4 Parity check: verify metric key set matches `docs/TRAINING_METRICS_AND_LOSSES.md`, with no per-dataset buckets, across a small matrix (packed vs non-packed; token-type metrics on/off; coord_soft_ce_w1 on/off)

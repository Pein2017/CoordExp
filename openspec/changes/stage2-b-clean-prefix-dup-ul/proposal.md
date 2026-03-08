## Why

Stage-2 Channel-B currently strict-parses the raw rollout, matches parsed objects, and builds the teacher-forced target from the raw accepted prefix plus FN append. In the duplicate-heavy failure mode, this leaves a cheap object-level escape hatch:

- near-duplicate bbox objects can repeat many times with the same normalized description,
- Hungarian can still salvage one matched localization,
- later correct objects are still teacher-forced under duplicate-contaminated prefixes,
- generic unmatched predictions remain FP-neutral, but duplicate-certified continuations are not distinguished from other unmatched extras.

The practical result is the known high-cardinality local optimum documented in the project notes: matched localization stays fairly stable, while prediction count, duplicate density, and truncation worsen.

We need a minimal Stage-2 Channel-B v2 contract that:

- preserves Stage-1, CoordExp, desc-first generation, and no-extra-objectness-head philosophy,
- keeps generic unmatched accepted objects FP-neutral by default,
- adds targeted negative signal only for duplicate-certified continuations,
- rebuilds Channel-B teacher forcing around a clean deduplicated prefix,
- keeps Channel-A behavior as stable as possible.

## What Changes

- Add a new Channel-B v2 training contract:
  - `raw rollout -> strict parse -> bbox-valid filtering -> sequential dedup -> clean accepted sequence + duplicate bursts -> Hungarian on clean accepted -> clean-prefix CE + duplicate UL`
- Formally allow Channel-B v2 to rebuild a deduplicated clean assistant target, superseding the older immutable-rollout-prefix rule for this path only.
- Add deterministic sequential dedup for bbox records using shared `normalize_desc` plus a configurable near-duplicate IoU threshold.
- Match GT against the clean accepted sequence, not the raw duplicate-heavy parsed list.
- Rebuild Channel-B positive supervision from the clean accepted sequence so later correct objects use clean prefixes rather than duplicate-contaminated prefixes.
- Add a new explicit Stage-2 objective atom for duplicate-start unlikelihood:
  - target only duplicate-certified continuations,
  - define the bad token as the first true LCP-divergence token relative to the clean continuation,
  - aggregate as one UL term per unique divergence token per clean boundary, with unit weight in v1.
- Extend diagnostics/config surfaces for duplicate collapse monitoring, including duplicate counts, burst counts, UL-application counts, and near-duplicate set metrics.
- Add a safer recommended Stage-2 v2 config that stays A-hot / B-cold and reduces overly permissive long-rollout behavior.

## Capabilities

### New Capabilities
- `stage2-b-clean-prefix-dup-ul`: Stage-2 Channel-B clean-prefix duplicate suppression with duplicate-only unlikelihood.

### Modified Capabilities
- `stage2-ab-training`: Channel-B matching and teacher-forced target construction move from raw-prefix semantics to clean-prefix semantics under the v2 contract.
- `teacher-forcing-unified-loss-registry`: add a new explicit duplicate UL objective atom for Stage-2 Channel-B without changing existing `token_ce` semantics.
- `trainer-metrics-components`: extend the metrics surface with duplicate-collapse diagnostics and boundary-level duplicate UL counters.

## Impact

- Default Stage-1 behavior is unchanged.
- Channel-A behavior remains unchanged except for small integration updates required to coexist with the new Channel-B atom/config plumbing.
- Stage-2 Channel-B becomes stricter only for duplicate-certified bbox continuations; generic unmatched accepted objects remain neutral.
- Matching, FN detection, and downstream positive CE become more semantically consistent because they operate on the clean accepted sequence.
- New configs, metrics, and tests make the duplicate-collapse behavior auditable before broader experimentation.

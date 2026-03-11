## Why

Stage-2 Channel-B currently strict-parses the raw rollout, bbox-filters the parsed records, matches directly on that raw filtered list, and builds the teacher-forced target from the raw accepted prefix plus FN append.

In the duplicate-heavy failure mode, that leaves a cheap object-level escape hatch:

- near-duplicate bbox objects can repeat many times with the same normalized description,
- Hungarian can still salvage one matched localization,
- later correct objects are still teacher-forced under duplicate-contaminated prefixes,
- generic unmatched predictions remain FP-neutral, but duplicate-certified continuations are not distinguished from other unmatched extras.

The practical result is the known high-cardinality local optimum documented in the project notes: matched localization stays fairly stable, while prediction count, duplicate density, and truncation worsen.

We need a Stage-2 Channel-B contract that:

- preserves Stage-1, CoordExp, desc-first generation, and no-extra-objectness-head philosophy,
- keeps generic unmatched accepted objects FP-neutral by default,
- adds targeted negative signal only for duplicate-certified continuations,
- rebuilds Channel-B teacher forcing around a clean deduplicated prefix,
- removes the old raw-prefix/immutable-prefix Channel-B contract rather than carrying compatibility baggage,
- keeps Channel-A behavior as stable as possible.

## What Changes

- Replace the Stage-2 two-channel Channel-B raw-prefix contract with a single canonical clean-prefix contract:
  - `raw rollout -> container salvage + strict record acceptance -> bbox-valid filtering -> sequential dedup -> clean accepted sequence + duplicate bursts -> Hungarian on clean accepted -> clean-prefix CE + duplicate UL`
- Remove the old immutable-rollout-prefix rule for Stage-2 two-channel Channel-B and do not provide a compatibility mode, version selector, or dedup enable/disable escape hatch.
- Make clean-prefix training semantics mandatory:
  - later correct objects are teacher-forced on the clean accepted prefix,
  - duplicates are removed from the positive teacher-forced sequence and reintroduced only as boundary-local duplicate-UL targets,
  - generic unmatched clean accepted objects remain neutral context.
- Establish a canonical config hierarchy:
  - `stage2_ab.channel_b.*` owns Channel-B rollout-prep invariants,
  - `stage2_ab.pipeline.objective[]` owns objective weights,
  - `duplicate_ul` is a first-class B-only objective module with no extra flat weight surface.
- Add a new explicit Stage-2 objective atom for duplicate unlikelihood:
  - target only duplicate-certified continuations,
  - define the bad token as the first true LCP-divergence token relative to the clean continuation,
  - aggregate as one UL term per unique divergence token per clean boundary,
  - intentionally collapse same-boundary duplicates that share the same divergence token so duplicate-burst length does not scale the loss by repeated identical bad continuations.
- Extend diagnostics/config surfaces for duplicate collapse monitoring, including duplicate gauges, duplicate/UL counters, and safer A-hot/B-cold recommended profiles.
- Add a docs-sync requirement after implementation:
  - update `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS.md`,
  - then review `docs/eval/README.md`, `docs/ARTIFACTS.md`, and `docs/README.md` for contract/index/artifact sync.

## Capabilities

### Modified Capabilities
- `stage2-ab-training`: replace Channel-B raw-prefix semantics with a canonical clean-prefix contract and typed hierarchy for dedup + duplicate UL.
- `teacher-forcing-objective-pipeline`: add `duplicate_ul` as a strict B-only objective module in the ordered pipeline surface.
- `teacher-forcing-unified-loss-registry`: define duplicate-only rollout semantics, clean-boundary divergence targeting, and duplicate UL behavior without changing existing positive `token_ce` semantics.
- `trainer-metrics-components`: extend the metrics surface with duplicate-collapse gauges, Channel-B duplicate/UL counters, and canonical logging keys for the new objective atom.

## Impact

- Default Stage-1 behavior is unchanged.
- Channel-A behavior remains unchanged except for small shared pipeline/config plumbing updates required to host the new Channel-B objective atom and metrics.
- Stage-2 two-channel Channel-B becomes a breaking contract change by design:
  - clean-prefix Channel-B is the only supported path after this change,
  - there is no backward-compatibility mode for the old raw-prefix contract.
- Matching, FN detection, and downstream positive CE become more semantically consistent because they operate on the clean accepted sequence.
- New configs, metrics, tests, and doc-sync tasks make the duplicate-collapse behavior auditable before broader experimentation.

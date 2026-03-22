# Roadmap: CoordExp Stage-2 Pseudo-Positive Implementation

## Overview

This roadmap turns the validated OpenSpec change `study-channel-b-pseudopositive-promotion` into a bounded brownfield implementation inside the current CoordExp Stage-2 trainer. The work is sequenced to preserve the current `K=2` baseline, generalize rollout evidence safely to arbitrary `K`, project pseudo-positive supervision through the existing one-forward coord-side path, and finish with compatibility-safe observability and validation that enables the next `best-K` study.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Config And Rollout Foundation** - Add the versionless pseudo-positive config contract and arbitrary-`K` runtime scaffolding without breaking the disabled path.
- [ ] **Phase 2: Support-Rate Triage And Promotion** - Turn multi-explorer evidence into deterministic support-rate buckets, recovered-FN aggregation, and cluster-safe pseudo-positive selection.
- [ ] **Phase 3: One-Forward Loss Projection** - Realize pseudo-positive coord-only supervision and duplicate-like dead-anchor suppression under the existing one-forward target contract.
- [ ] **Phase 4: Observability And Operator Surface** - Add metrics, metadata, docs, and YAML authoring surfaces that keep arbitrary-`K` runs interpretable and backward-compatible.
- [ ] **Phase 5: Validation And Best-K Readiness** - Prove the contract with targeted tests, smoke validation, and first `best-K`-ready comparison support.

## Phase Details

### Phase 1: Config And Rollout Foundation
**Goal**: Make pseudo-positive mode legal, deterministic, and baseline-safe before any triage or loss behavior changes land.
**Depends on**: Nothing (first phase)
**Requirements**: [CONF-01, CONF-02, CONF-03, ROLL-01, ROLL-02, ROLL-03, ROLL-04, ROLL-05]
**Success Criteria** (what must be TRUE):
  1. Operator-authored YAML can enable pseudo-positive mode only through versionless `stage2_ab.channel_b.pseudo_positive.*` keys, and invalid keys/weights fail fast.
  2. Enabled Channel-B runtime schedules `1` anchor plus `num_rollouts - 1` explorers with deterministic explorer identity and unchanged disabled-path behavior.
  3. Malformed anchor-preparation samples are dropped, malformed explorer-preparation results abort the step, and zero-object explorers remain valid zero-support evidence.
**Plans**: 3 plans

Plans:
- [ ] 01-01: Extend config schema, defaults, and guardrail tests for `pseudo_positive.*`
- [ ] 01-02: Generalize Channel-B rollout scheduling from one explorer to arbitrary `K-1` explorers
- [ ] 01-03: Implement fixed-denominator failure handling and disabled-path compatibility checks

### Phase 2: Support-Rate Triage And Promotion
**Goal**: Implement the core pseudo-positive selection algorithm inside `target_builder.py`.
**Depends on**: Phase 1
**Requirements**: [TRIA-01, TRIA-02, TRIA-03, TRIA-04, TRIA-05]
**Success Criteria** (what must be TRUE):
  1. Each unmatched anchor object records deterministic explorer support count and support rate from arbitrary-`K` evidence using the geometry-first gates.
  2. Recovered GT uses any-hit collapse across explorers and yields at most one recovered FN object per missed GT object.
  3. Pseudo-positive promotion requires `support_count >= 2` and `support_rate >= 2/3`, and overlap clustering keeps exactly one promoted winner per duplicate-like candidate cluster.
**Plans**: 3 plans

Plans:
- [ ] 02-01: Add per-explorer anchor association, support-count/rate accounting, and recovered-FN aggregation
- [ ] 02-02: Implement support-rate bucket assignment for `dead_anchor`, `shielded_anchor`, and pseudo-positive candidates
- [ ] 02-03: Add connected-component clustering with deterministic winner selection and demotion

### Phase 3: One-Forward Loss Projection
**Goal**: Route pseudo-positive anchors into coord-only supervision and keep dead-anchor negatives narrow without breaking the one-forward contract.
**Depends on**: Phase 2
**Requirements**: [LOSS-01, LOSS-02, LOSS-03, LOSS-04]
**Success Criteria** (what must be TRUE):
  1. Selected pseudo-positive anchors appear in coord-side bbox groups using anchor-owned target coordinates and never create desc CE or matched-prefix structure CE.
  2. The trainer still runs one edited-anchor teacher-forced forward for the whole sample.
  3. Dead anchors remain outside the final target, and only duplicate-like dead branches emit first-divergence suppression targets.
**Plans**: 3 plans

Plans:
- [ ] 03-01: Thread pseudo-positive winners into bbox-group creation with coord-only weighting
- [ ] 03-02: Verify objective modules consume pseudo-positive groups only on coord-side paths
- [ ] 03-03: Filter dead-anchor suppression targets to duplicate-like local branches only

### Phase 4: Observability And Operator Surface
**Goal**: Make arbitrary-`K` pseudo-positive runs measurable, comparable, and operable without breaking existing metric meaning.
**Depends on**: Phase 3
**Requirements**: [OBS-01, OBS-02, OBS-03, OBS-04, OBS-05]
**Success Criteria** (what must be TRUE):
  1. Enabled pseudo-positive runs emit the required per-sample metadata fields and documented `train/triage/*` aggregates.
  2. `train/triage/unlabeled_consistent_count` and legacy `rollout/explorer/*` metrics retain their documented compatibility meaning under arbitrary `K`.
  3. Operators can find the new YAML, metric, and failure semantics in `docs/training/METRICS.md` and `docs/training/STAGE2_RUNBOOK.md`.
**Plans**: 3 plans

Plans:
- [ ] 04-01: Add per-sample metadata carriers and aggregate pseudo-positive metrics
- [ ] 04-02: Preserve compatibility semantics for legacy triage and explorer metrics
- [ ] 04-03: Update operator docs and author an explicit pseudo-positive YAML profile

### Phase 5: Validation And Best-K Readiness
**Goal**: Prove the implementation is correct, regression-safe, and ready for the first `best-K` ablation study.
**Depends on**: Phase 4
**Requirements**: [VAL-01, VAL-02, VAL-03, VAL-04]
**Success Criteria** (what must be TRUE):
  1. Config-contract and Stage-2 trainer regression suites pass for the new pseudo-positive surface.
  2. A smoke-ready YAML profile exercises the enabled path and can be compared across at least two enabled `num_rollouts` values.
  3. Enabled `K=2` behaves as a no-promotion control and does not accidentally create pseudo-positive winners.
**Plans**: 3 plans

Plans:
- [ ] 05-01: Add and pass targeted config-contract and trainer regression tests
- [ ] 05-02: Run pseudo-positive smoke validation and confirm one-forward / coord-only behavior
- [ ] 05-03: Verify `best-K`-ready observability and `K=2` no-promotion control semantics

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Config And Rollout Foundation | 0/3 | Not started | - |
| 2. Support-Rate Triage And Promotion | 0/3 | Not started | - |
| 3. One-Forward Loss Projection | 0/3 | Not started | - |
| 4. Observability And Operator Surface | 0/3 | Not started | - |
| 5. Validation And Best-K Readiness | 0/3 | Not started | - |

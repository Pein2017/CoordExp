# Project Research Summary

**Project:** CoordExp Stage-2 Pseudo-Positive Implementation
**Domain:** Brownfield implementation of arbitrary-K pseudo-positive Channel-B training
**Researched:** 2026-03-22
**Confidence:** HIGH

## Executive Summary

This project is not a greenfield product build. It is a tightly scoped brownfield feature implementation inside the existing CoordExp Stage-2 trainer, governed by the OpenSpec change `study-channel-b-pseudopositive-promotion`. The research outcome is therefore intentionally implementation-centric: reuse the current `stage2_two_channel` trainer/runtime, keep the edited-anchor one-forward architecture, and extend only the exact authority seams needed for arbitrary-`K` explorer evidence, support-rate pseudo-positive selection, coord-only pseudo-positive losses, and compatibility-safe observability.

The recommended approach is to stage the work in the same order that the runtime depends on it. First lock the config contract and establish `K=4` as the default pseudo-positive rollout profile while preserving the legacy `K=2` path. Then implement support-count / support-rate triage and overlap clustering in `target_builder.py`. After that, wire selected pseudo-positive anchors into the existing coord-side bbox-group path without leaking into text-side losses, and preserve dead-anchor suppression as a narrow duplicate-like first-divergence penalty. Finish with observability, YAML profiles, docs, and smoke validation so the feature is both implementable and ablation-ready.

The critical risks are not “can the repo support this?” but “can we add it without making the current trainer ambiguous?” The main pitfalls are denominator drift when explorers fail, accidental text-side pseudo-positive supervision, duplicate-like over-promotion, and silently redefining existing metrics. The roadmap should therefore group work around contract guardrails first, algorithmic triage second, loss realization third, and compatibility/validation last.

## Key Findings

### Recommended Stack

The correct stack is the repo’s current Stage-2 runtime, not a new experimental branch. The implementation should stay within:

- `src/config/schema.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/target_builder.py`
- `src/trainers/stage2_two_channel/rollout_views.py`
- `src/trainers/rollout_matching/matching.py`
- `src/trainers/teacher_forcing/modules/*`

Validation should use the existing harness:

- `openspec validate study-channel-b-pseudopositive-promotion --type change --strict --json --no-interactive`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
- `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`

**Core technologies / surfaces:**
- `schema.py`: typed pseudo-positive config and invariants
- `stage2_two_channel.py`: arbitrary-`K` rollout scheduling, batch metrics, failure policy
- `target_builder.py`: support/rate triage, clustering, final target, dead suppression targets
- teacher-forcing bbox modules: coord-only pseudo-positive loss projection

### Expected Features

The table-stakes implementation features are:

- versionless typed `stage2_ab.channel_b.pseudo_positive.*` config
- arbitrary-`K` rollout preparation with `1` anchor + `K-1` explorers
- support-count / support-rate pseudo-positive triage
- connected-component overlap clustering for pseudo-positive winners
- coord-only pseudo-positive loss wiring
- duplicate-like dead-anchor suppression only
- compatibility-safe metrics, docs, and YAML profiles

**Must have (table stakes):**
- typed pseudo-positive config and guards
- arbitrary-`K` explorer scheduling and fixed-denominator handling
- support-rate triage and recovered-GT collapse
- coord-only pseudo-positive groups under one forward
- metrics/doc compatibility and smoke validation

**Should have (competitive / next-stage):**
- broader `best-K` ablation coverage
- richer explorer-local diagnostics

**Defer (v2+):**
- semantic-desc gating
- explorer-only pseudo-positive paths
- dynamic pseudo-positive weighting schedules

### Architecture Approach

The architecture should stay anchor-owned and evidence-driven. `stage2_two_channel.py` schedules the views and aggregates metrics. `rollout_views.py` prepares accepted-clean objects independently per view. `target_builder.py` turns those views into support rates, bucket assignments, overlap-cluster winners, recovered FN objects, bbox groups, and dead-anchor suppression targets. The teacher-forcing pipeline consumes that single clean target and projects losses through existing modules without adding a second forward or widening text-side supervision.

**Major components:**
1. `schema.py` — config legality and invariants
2. `stage2_two_channel.py` — runtime orchestration, explorer aggregation, failure semantics
3. `target_builder.py` — algorithmic triage and target construction
4. `teacher_forcing/modules/*` — loss realization
5. `docs/training/*` + tests — stable operator contract and regression coverage

### Critical Pitfalls

1. **Silent explorer-denominator drift** — hard-abort malformed explorer prep; never shrink denominators silently.
2. **Pseudo-positive leakage into text supervision** — restrict pseudo-positive to bbox-group-based coord modules only.
3. **Over-promoting near-duplicate unmatched anchors** — cluster overlapping candidates and keep only one winner.
4. **Breaking legacy metrics** — preserve `unlabeled_consistent_count` and define legacy explorer summaries explicitly.
5. **Regressing legacy `K=2` behavior** — keep all new behavior behind `pseudo_positive.enabled`, use `K=4` as the default pseudo-positive profile, and retain enabled `K=2` as a no-promotion control.

## Implications for Roadmap

Based on the research, suggested phase structure:

### Phase 1: Config And Runtime Foundation
**Rationale:** The rest of the feature is unsafe until the YAML/schema surface and arbitrary-`K` runtime skeleton are legal and deterministic.
**Delivers:** Typed config, invariants, arbitrary-`K` rollout scheduling, anchor/explorer failure policy.
**Addresses:** config contract, rollout scheduling table stakes.
**Avoids:** disabled-path regressions and denominator drift.

### Phase 2: Support-Rate Triage And Promotion
**Rationale:** Once multiple explorer views exist, the core algorithmic step is turning them into pseudo-positive decisions.
**Delivers:** support counts/rates, recovered-GT collapse, promotion thresholds, overlap clustering.
**Uses:** existing matching/view-prep surfaces.
**Implements:** target-builder triage and selection logic.

### Phase 3: One-Forward Loss Projection
**Rationale:** With buckets defined, loss wiring can be implemented cleanly and tested precisely.
**Delivers:** coord-only pseudo-positive bbox groups, anchor-owned targets, duplicate-like dead-anchor filtering.
**Avoids:** text-side leakage and full-object dead CE.

### Phase 4: Observability And Operator Surface
**Rationale:** Metrics, metadata, YAML profiles, and docs should be derived from the actual runtime behavior.
**Delivers:** aggregate metrics, per-sample metadata, YAML profile(s), runbook and metrics docs updates.
**Implements:** compatibility-safe `best-K` observability.

### Phase 5: Validation And Best-K Readiness
**Rationale:** The feature is only useful if it can be verified and compared across `K`.
**Delivers:** targeted tests, smoke validation, and first `best-K` comparison readiness.
**Addresses:** regression risk and research usefulness.

### Phase Ordering Rationale

- Config and rollout scheduling must exist before support-rate triage.
- Triage must exist before bbox groups and dead-anchor filtering can be correct.
- Metrics/docs should follow final runtime semantics, not precede them.
- Validation comes last because it needs the complete integrated behavior.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** highest algorithmic concentration; bucket semantics and clustering details need careful test design.
- **Phase 3:** easiest place to accidentally violate the one-forward or coord-only contract.

Phases with standard patterns:
- **Phase 1:** largely schema/runtime extension work with strong existing authority surfaces.
- **Phase 4:** mostly compatibility/documentation work once semantics are fixed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Authority surfaces and verification handles are clear in the repo docs and change tasks. |
| Features | HIGH | The OpenSpec change and tasks already bound the v1 surface tightly. |
| Architecture | HIGH | The one-forward, anchor-owned runtime is stable and explicitly documented. |
| Pitfalls | HIGH | Most risks are concrete compatibility and data-flow issues already visible in the existing code/spec shape. |

**Overall confidence:** HIGH

### Gaps To Address

- The exact plan/task decomposition inside each phase still needs to be authored in GSD phase plans.
- A small number of runtime details, such as the minimal YAML profile choice and the exact smoke config, should be finalized during phase planning rather than here.

---
*Research summary for: CoordExp Stage-2 pseudo-positive implementation*
*Researched: 2026-03-22*

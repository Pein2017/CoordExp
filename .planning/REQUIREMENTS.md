# Requirements: CoordExp Stage-2 Pseudo-Positive Implementation

**Defined:** 2026-03-22
**Core Value:** Ship the exact OpenSpec-defined arbitrary-`K` pseudo-positive Channel-B implementation correctly, reproducibly, and with backward-compatible safety around the existing `K=2` trainer path.

## v1 Requirements

### Config Contract

- [ ] **CONF-01**: Operator can enable pseudo-positive Channel-B behavior through `stage2_ab.channel_b.pseudo_positive.enabled` in YAML without adding any new CLI flags.
- [ ] **CONF-02**: Config loading rejects unknown nested pseudo-positive keys, rejects versioned `v*` knob aliases, and enforces `0.0 < coord_weight < 1.0`.
- [ ] **CONF-03**: Disabled pseudo-positive configs preserve canonical existing `K=2` Stage-2 behavior without changing current prod or smoke config semantics.

### Rollout Runtime

- [ ] **ROLL-01**: When pseudo-positive mode is enabled, the trainer schedules exactly `1` deterministic anchor rollout plus `num_rollouts - 1` explorer rollouts for each Channel-B sample.
- [ ] **ROLL-02**: Explorer rollouts reuse the same authored decode profile and differ only by deterministic explorer identity derived from the rollout seed base plus explorer ordinal.
- [ ] **ROLL-03**: A malformed anchor-preparation sample is dropped from Channel-B training for that step rather than using the legacy empty-prefix fallback.
- [ ] **ROLL-04**: A malformed or missing explorer-preparation result aborts the current optimizer step instead of silently shrinking the support-rate denominator.
- [ ] **ROLL-05**: Zero accepted-clean explorer output remains valid and contributes zero support rather than triggering fallback behavior.

### Triage And Promotion

- [ ] **TRIA-01**: The trainer computes per-anchor explorer support counts and support rates using the existing geometry-first unmatched and GT-conflict gates.
- [ ] **TRIA-02**: The trainer collapses recovered GT across explorers so each anchor-missed GT object creates at most one recovered FN object in the final supervision contract.
- [ ] **TRIA-03**: The trainer promotes an unmatched anchor object to pseudo-positive status only when it satisfies both `support_count >= 2` and `support_rate >= 2/3`.
- [ ] **TRIA-04**: The trainer keeps unsupported unmatched anchors as `dead_anchor` and partially supported unmatched anchors as `shielded_anchor` rather than treating all unmatched anchors the same.
- [ ] **TRIA-05**: Overlapping pseudo-positive candidates are clustered deterministically, with exactly one promoted winner per cluster and all non-winners demoted back to `shielded_anchor`.

### Objective And Loss Semantics

- [ ] **LOSS-01**: Selected pseudo-positive anchors contribute only coord-side supervision through anchor-owned target coordinates and never create desc CE or matched-prefix structure CE.
- [ ] **LOSS-02**: Pseudo-positive `coord_weight` scales only the coord-side modules `bbox_geo`, `coord_reg`, and `bbox_size_aux`.
- [ ] **LOSS-03**: The trainer preserves one edited-anchor teacher-forced forward for the whole sample and does not add a second teacher trajectory for pseudo-positive supervision.
- [ ] **LOSS-04**: Dead anchors stay out of the final target, and only duplicate-like dead branches create explicit first-divergence suppression targets.

### Observability And Operator Surface

- [ ] **OBS-01**: Per-sample Channel-B metadata records valid explorer count, per-anchor support counts/rates, pseudo-positive selections, dead explorers by view, and recovered-GT support counts/rates whenever pseudo-positive mode is enabled.
- [ ] **OBS-02**: Aggregate metrics expose pseudo-positive counts and rate numerators/denominators in the `train/triage/*` namespace without breaking existing metric families.
- [ ] **OBS-03**: `train/triage/unlabeled_consistent_count` remains the total shielded-anchor count under pseudo-positive mode for backward compatibility.
- [ ] **OBS-04**: Legacy `rollout/explorer/*` metrics remain defined under arbitrary `K` as deterministic mean-over-valid-explorer-view summaries.
- [ ] **OBS-05**: Operator docs describe the new pseudo-positive YAML knobs, metric meanings, and failure semantics in `docs/training/METRICS.md` and `docs/training/STAGE2_RUNBOOK.md`.

### Validation And Ablation Readiness

- [ ] **VAL-01**: Targeted config-contract tests prove the pseudo-positive schema, invariants, and compatibility guards work as authored.
- [ ] **VAL-02**: Targeted Stage-2 trainer tests prove arbitrary-`K` rollout prep, support-rate triage, one-forward coord-only pseudo-positive losses, and duplicate-like dead suppression behavior.
- [ ] **VAL-03**: An explicit pseudo-positive YAML profile can be loaded and exercised in a smoke-ready path with at least two enabled `num_rollouts` values for `best-K` comparison.
- [ ] **VAL-04**: Enabled `K=2` runs behave as a no-promotion control under the `support_count >= 2` floor.

## v2 Requirements

### Future Research

- **FUTR-01**: Pseudo-positive selection may incorporate semantic-desc agreement or semantic vote weighting after the geometry-first implementation is stable.
- **FUTR-02**: Pseudo-positive weighting may vary by support rate or confidence band instead of using one global `coord_weight`.
- **FUTR-03**: Explorer-only non-anchor pseudo-positive proposals may be studied after anchor-centric v1 behavior is validated.
- **FUTR-04**: Broader `best-K` ablation automation may be added once the first two-point comparison path is stable.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Pseudo-positive desc CE | Explicitly deferred by the authored OpenSpec change; too risky for v1. |
| Pseudo-positive matched-prefix structure CE | Explicitly deferred; v1 keeps text-side supervision unchanged outside matched/FN paths. |
| Full-object negative CE for dead anchors | Violates the conservative dead-anchor contract and risks over-penalizing incompletely labeled scenes. |
| New CLI flags for pseudo-positive behavior | Repo guardrails require YAML-first configuration. |
| Broad Stage-2 architecture redesign | The project is scoped to the exact OpenSpec implementation slice only. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CONF-01 | TBD | Pending |
| CONF-02 | TBD | Pending |
| CONF-03 | TBD | Pending |
| ROLL-01 | TBD | Pending |
| ROLL-02 | TBD | Pending |
| ROLL-03 | TBD | Pending |
| ROLL-04 | TBD | Pending |
| ROLL-05 | TBD | Pending |
| TRIA-01 | TBD | Pending |
| TRIA-02 | TBD | Pending |
| TRIA-03 | TBD | Pending |
| TRIA-04 | TBD | Pending |
| TRIA-05 | TBD | Pending |
| LOSS-01 | TBD | Pending |
| LOSS-02 | TBD | Pending |
| LOSS-03 | TBD | Pending |
| LOSS-04 | TBD | Pending |
| OBS-01 | TBD | Pending |
| OBS-02 | TBD | Pending |
| OBS-03 | TBD | Pending |
| OBS-04 | TBD | Pending |
| OBS-05 | TBD | Pending |
| VAL-01 | TBD | Pending |
| VAL-02 | TBD | Pending |
| VAL-03 | TBD | Pending |
| VAL-04 | TBD | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 0
- Unmapped: 26 ⚠️

---
*Requirements defined: 2026-03-22*
*Last updated: 2026-03-22 after brownfield research synthesis*

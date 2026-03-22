# Architecture Research

**Domain:** Brownfield Stage-2 pseudo-positive training architecture in CoordExp
**Researched:** 2026-03-22
**Confidence:** HIGH

## Standard Architecture

### System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Config / Bootstrap                                                  │
│  schema.py -> trainer_setup.py -> pipeline manifest / runtime setup │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│ Stage-2 Runtime                                                     │
│  stage2_two_channel.py                                              │
│  - schedule anchor + explorers                                      │
│  - prepare per-view rollout views                                   │
│  - aggregate triage / metrics / failure handling                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│ Target Construction                                                 │
│  target_builder.py                                                  │
│  - support counts / rates                                           │
│  - recovered FN collapse                                            │
│  - overlap clustering                                               │
│  - final edited target + bbox groups + dead targets                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│ Teacher-Forcing Objective Pipeline                                  │
│  objective_runner.py + teacher_forcing/modules/*                    │
│  - token_ce                                                         │
│  - bbox_geo / bbox_size_aux / coord_reg                             │
│  - loss_dead_anchor_suppression                                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│ Metrics / Artifacts / Docs                                          │
│  rollout metrics, train/triage/*, runbook + metrics docs            │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `src/config/schema.py` | Legal config surface, defaults, invariants, unknown-key rejection | Add `stage2_ab.channel_b.pseudo_positive` and arbitrary-`K` validation rules. |
| `src/trainers/stage2_two_channel.py` | Per-step orchestration, rollout scheduling, batch-level metrics and failure policy | Generalize from one explorer to `K-1` explorers and define aggregate observability semantics. |
| `src/trainers/stage2_two_channel/rollout_views.py` | Per-view accepted-clean preparation | Reuse unchanged logic independently for each explorer view. |
| `src/trainers/stage2_two_channel/target_builder.py` | Triage, pseudo-positive promotion, bbox-group construction, dead-anchor suppression targets | Main algorithm surface for support-rate logic and one-forward target realization. |
| `src/trainers/teacher_forcing/modules/*` | Module-level losses and diagnostics | Keep text-side modules unchanged except where masks/groups now include pseudo-positive coord groups. |
| `docs/training/METRICS.md` + `docs/training/STAGE2_RUNBOOK.md` | Stable operator-facing documentation | Update only after metrics and YAML profile semantics are concrete in code/tests. |

## Recommended Project Structure

```text
src/
├── config/
│   └── schema.py                         # typed pseudo_positive config + invariants
├── trainers/
│   ├── stage2_two_channel.py             # arbitrary-K rollout scheduling + aggregate metrics
│   ├── stage2_two_channel/
│   │   ├── rollout_views.py              # unchanged per-view accepted-clean prep reuse
│   │   ├── target_builder.py             # support/rate triage, clustering, bbox groups, dead targets
│   │   ├── objective_runner.py           # routes built targets to objective modules
│   │   └── types.py                      # metadata carriers if new fields are required
│   └── teacher_forcing/
│       └── modules/
│           ├── bbox_geo.py               # weighted coord-side supervision
│           ├── bbox_size_aux.py          # optional size auxiliary on pseudo-positive groups
│           ├── coord_reg.py              # coord-side regularization
│           └── loss_dead_anchor_suppression.py
configs/
├── stage2_two_channel/
│   ├── base.yaml
│   └── prod/ or smoke/                   # explicit pseudo_positive-enabled profile
tests/
├── test_stage2_ab_config_contract.py
└── test_stage2_ab_training.py
docs/
└── training/
    ├── METRICS.md
    └── STAGE2_RUNBOOK.md
```

### Structure Rationale

- **`src/config/`:** config invariants must fail early before runtime changes are exercised.
- **`src/trainers/stage2_two_channel.py`:** rollout scheduling and batch-level observability must stay centralized so legacy metric meaning remains consistent.
- **`src/trainers/stage2_two_channel/target_builder.py`:** all promotion, demotion, grouping, and dead-target decisions should live in one place to preserve one-forward reasoning.
- **`src/trainers/teacher_forcing/modules/`:** existing loss atoms should be reused rather than replaced, with only the group membership and weights extended.

## Architectural Patterns

### Pattern 1: Anchor-Owned Final Target

**What:** Build one edited clean target from the anchor rollout only, then teacher-force that single target.
**When to use:** Always for this feature slice.
**Trade-offs:** Preserves stability and compatibility, but limits pseudo-positive scope to anchor-side objects.

**Example:**
```text
anchor + explorers -> triage/support -> edited anchor target -> one teacher-forced forward
```

### Pattern 2: Evidence-Only Explorers

**What:** Treat explorer rollouts as evidence probes rather than alternate training trajectories.
**When to use:** For arbitrary-`K` pseudo-positive selection and recovered-GT evidence.
**Trade-offs:** Clean denominator semantics and easier reproducibility, but no explorer-only pseudo-positive objects in v1.

### Pattern 3: Weight-Carried Coord Projection

**What:** Reuse the existing bbox-group weight carrier to scale pseudo-positive coord supervision without adding new global optimizer weights.
**When to use:** For `bbox_geo`, `coord_reg`, and `bbox_size_aux`.
**Trade-offs:** Minimal implementation surface and clear module targeting, but only one coarse weight in v1.

## Data Flow

### Training Flow

```text
resolved config
    ↓
schema validation (`pseudo_positive.enabled`, `coord_weight`, `num_rollouts`)
    ↓
stage2_two_channel rollout scheduling
    ↓
anchor view prep + explorer view prep (per-view accepted-clean pipeline)
    ↓
target_builder:
  - anchor/explorer association per explorer
  - support_count / support_rate
  - recovered_fn collapse
  - overlap clustering
  - final edited target
  - bbox groups / text masks / dead suppression targets
    ↓
objective pipeline:
  - token_ce sees matched-prefix + FN text targets only
  - bbox_geo / coord_reg / bbox_size_aux see matched + FN + pseudo-positive coord groups
  - dead_anchor_suppression sees duplicate-like dead-branch targets only
    ↓
aggregate metrics + per-sample metadata
```

### Key Data Flows

1. **Rollout evidence flow:** `num_rollouts` config -> `stage2_two_channel.py` scheduling -> per-view `rollout_views.py` prep -> `target_builder.py` support accounting.
2. **Pseudo-positive supervision flow:** selected anchor indices -> bbox-group creation in `target_builder.py` -> weighted coord-side modules in the objective pipeline.
3. **Dead suppression flow:** dead-anchor boundary groups -> duplicate-like filter -> first-divergence bad-token targets -> `loss_dead_anchor_suppression.py`.
4. **Observability flow:** per-view prep + triage meta -> batch aggregation in `stage2_two_channel.py` -> `train/triage/*`, `rollout/explorer/*`, and optional monitoring payloads.

## Suggested Build Order

1. **Config contract first**
   Why: invalid YAML or bad invariants should fail before any runtime changes land.
2. **Arbitrary-`K` rollout scheduling and per-view preparation**
   Why: support-rate triage depends on stable explorer view materialization.
3. **Support/rate triage + overlap clustering + recovered-GT aggregation**
   Why: this is the central algorithmic layer that defines every downstream bucket.
4. **Coord-side loss wiring + dead-anchor filtering**
   Why: once buckets exist, the objective pipeline can project them without redesigning the forward path.
5. **Metrics/docs/YAML profile + smoke validation**
   Why: observability and operator docs should be derived from the final runtime behavior, not guessed early.

## Highest-Risk Interfaces

- **`stage2_two_channel.py` <-> `target_builder.py`:** singular-explorer assumptions must be removed without breaking disabled-path behavior.
- **`target_builder.py` <-> teacher-forcing bbox groups:** pseudo-positive groups need anchor-owned targets and coord-only weight scaling with no text leakage.
- **runtime failures <-> metrics:** explorer-prep hard aborts cannot be promised as ordinary finalized step metrics, so telemetry/reporting needs careful treatment.
- **compatibility metrics <-> new semantics:** `unlabeled_consistent_count` and legacy `rollout/explorer/*` keys must remain interpretable under arbitrary `K`.

## Sources

- `/data/CoordExp/docs/IMPLEMENTATION_MAP.md`
- `/data/CoordExp/docs/training/STAGE2_RUNBOOK.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/.planning/PROJECT.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/specs/stage2-ab-training/spec.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/specs/teacher-forcing-unified-loss-registry/spec.md`

---
*Architecture research for: CoordExp Stage-2 pseudo-positive implementation*
*Researched: 2026-03-22*

# Stack Research

**Domain:** Brownfield implementation of arbitrary-K Stage-2 Channel-B pseudo-positive training in CoordExp
**Researched:** 2026-03-22
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Existing CoordExp Stage-2 trainer stack | repo-managed | Primary implementation surface | The change is explicitly bounded to the current Stage-2 authority files and should reuse the current one-forward training architecture rather than introducing a parallel runtime. |
| OpenSpec change `study-channel-b-pseudopositive-promotion` | authored delta | Normative contract source | The implementation is governed by the authored spec deltas for `stage2-ab-training`, `teacher-forcing-unified-loss-registry`, and `trainer-metrics-components`; this is the only safe contract source for the feature slice. |
| Conda environment `ms` + existing Python test harness | repo-managed | Verification surface | The repo already standardizes validation through `PYTHONPATH=. conda run -n ms python -m pytest ...`, so the implementation should stay inside that harness instead of inventing separate validation tooling. |

### Supporting Libraries / Subsystems

| Library / Subsystem | Version | Purpose | When to Use |
|---------------------|---------|---------|-------------|
| `src/config/schema.py` | repo-managed | Typed config validation for `stage2_ab.channel_b.pseudo_positive.*` and arbitrary-`K` guards | Extend first, because the rest of the implementation depends on a stable config contract. |
| `src/trainers/stage2_two_channel.py` | repo-managed | Rollout scheduling, per-step observability, and batch-level failure handling | Use to generalize from one explorer to `num_rollouts - 1` explorers while preserving one clean teacher-forced forward. |
| `src/trainers/stage2_two_channel/target_builder.py` | repo-managed | Triage, support accounting, pseudo-positive selection, dead-anchor suppression target building | This is the main algorithmic implementation surface for support-rate voting, overlap clustering, and duplicate-like dead-anchor filtering. |
| `src/trainers/stage2_two_channel/rollout_views.py` | repo-managed | Accepted-clean object preparation per rollout view | Reuse unchanged semantics per view instead of inventing a second parsing pipeline. |
| `src/trainers/rollout_matching/matching.py` | repo-managed | Deterministic one-to-one IoU matching | Reuse for anchor/explorer counterpart association and keep geometry-first gating aligned with the current contract. |
| `src/trainers/teacher_forcing/modules/bbox_geo.py` | repo-managed | Primary bbox/coord geo loss | Pseudo-positive coord supervision must flow through the existing per-bbox-group weight carrier here. |
| `src/trainers/teacher_forcing/modules/coord_reg.py` | repo-managed | Coord-side regularization | Keep pseudo-positive objects coord-only by reusing this existing coord-side module instead of introducing text-side atoms. |
| `src/trainers/teacher_forcing/modules/bbox_size_aux.py` | repo-managed | Optional light oversize regularization on supervised coord groups | Reuse as a harness-level auxiliary for pseudo-positive groups instead of building a separate oversized-box subsystem. |
| `src/trainers/teacher_forcing/modules/loss_dead_anchor_suppression.py` | repo-managed | First-divergence dead-branch suppression | Keep the existing branch-entry penalty mechanism and narrow it to duplicate-like dead anchors only. |
| `tests/test_stage2_ab_config_contract.py` | repo-managed | Config contract regression coverage | Use for all pseudo-positive config schema and unknown-key guardrails. |
| `tests/test_stage2_ab_training.py` | repo-managed | Stage-2 behavior regression coverage | Use for arbitrary-`K` rollout prep, triage, one-forward loss realization, and metrics compatibility checks. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `openspec validate study-channel-b-pseudopositive-promotion --type change --strict --json --no-interactive` | Keep implementation aligned to the authored change | Run before implementation and again after the code lands. |
| `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py` | Validate schema and config contract | Fastest targeted regression loop for new YAML/config semantics. |
| `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py` | Validate trainer behavior and observability | Main regression suite for rollout prep, triage, and loss-path correctness. |

## Installation / Setup

```bash
# work in the isolated implementation worktree
cd /data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals

# validate the authored change before implementation
openspec validate study-channel-b-pseudopositive-promotion --type change --strict --json --no-interactive

# targeted verification harness
PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py
PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Extend `stage2_two_channel` runtime directly | Build a parallel trainer variant for pseudo-positive experiments | Only if the existing trainer architecture becomes irreconcilable with the authored spec, which the current design does not indicate. |
| Reuse per-bbox-group weights for pseudo-positive coord supervision | Add new objective-module-wide pseudo-positive weights | Only if future work requires independent optimizer-level scaling outside bbox groups; v1 does not. |
| Reuse existing accepted-clean parsing and matching per explorer view | Build a specialized explorer-only parsing path | Only if later research proves the current view preparation is insufficient for arbitrary-`K`, which is not the v1 contract. |

## What NOT To Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `src/trainers/stage2_ab_training.py` and `src/trainers/stage2_ab/executors.py` as implementation authority | They are compatibility shims, not the Stage-2 authority surfaces for the post-refactor trainer | Use `src/trainers/stage2_two_channel.py` and `src/trainers/stage2_two_channel/*`. |
| New CLI flags for pseudo-positive control | Violates repo guardrails and creates an unnecessary parallel configuration surface | Author YAML-only knobs under `stage2_ab.channel_b.pseudo_positive.*`. |
| Text-side pseudo-positive supervision in v1 | The authored change explicitly keeps pseudo-positive supervision coord-only and defers desc/structure CE | Use existing matched/FN text supervision only. |
| Reinterpreting explorer failures as reduced denominators | Breaks rate comparability across `K` and undermines reproducibility | Keep fixed-denominator semantics and hard-abort explorer-prep failures. |
| A second teacher-forced forward or union-order target | Violates the current one-forward Stage-2 architecture | Keep one edited anchor target and one teacher-forced forward. |

## Stack Patterns By Variant

**If `pseudo_positive.enabled=false`:**
- Keep legacy `K=2` Stage-2 behavior available unchanged
- Because backward compatibility is a hard requirement and existing prod/smoke configs must remain valid

**If `pseudo_positive.enabled=true` and `num_rollouts >= 2`:**
- Use `1` anchor + `num_rollouts - 1` explorers with aggregate explorer observability
- Default the authored pseudo-positive profile to `num_rollouts=4`
- Because the spec turns repeated explorer evidence into support-rate-based triage while keeping the anchor-owned target

## Version Compatibility

| Surface | Compatible With | Notes |
|---------|-----------------|-------|
| `stage2_ab.channel_b.pseudo_positive.*` | existing `stage2_two_channel` YAMLs when disabled | Disabled path must preserve current behavior and reject versioned `v*` aliases. |
| Arbitrary-`K` rollout prep | legacy `rollout/explorer/*` metrics | Legacy explorer metrics must remain defined as mean-over-valid-explorer-view summaries. |
| `train/triage/unlabeled_consistent_count` | new pseudo-positive support-rate metrics | Must remain the total shielded-anchor count for compatibility, not just the subthreshold subset. |

## Sources

- `/data/CoordExp/docs/PROJECT_CONTEXT.md` — precedence and documentation responsibilities
- `/data/CoordExp/docs/IMPLEMENTATION_MAP.md` — canonical Stage-2 implementation routing
- `/data/CoordExp/docs/training/STAGE2_RUNBOOK.md` — active Stage-2 runtime contract and validation handles
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/.planning/PROJECT.md` — scoped brownfield implementation context
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/proposal.md` — change motivation and constraints
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/tasks.md` — implementation-authority checklist

---
*Stack research for: CoordExp Stage-2 pseudo-positive implementation*
*Researched: 2026-03-22*

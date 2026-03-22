# CoordExp Stage-2 Pseudo-Positive Implementation

## What This Is

This project is the implementation planning slice for the OpenSpec change `study-channel-b-pseudopositive-promotion` inside the current CoordExp worktree. It exists to land the exact arbitrary-`K` Stage-2 Channel-B pseudo-positive contract, tests, observability, and smoke validation in the existing brownfield training stack, with `K=4` as the default pseudo-positive rollout profile and the legacy `K=2` path retained for compatibility and control comparisons.

## Core Value

Ship the exact OpenSpec-defined arbitrary-`K` pseudo-positive Channel-B implementation correctly, reproducibly, with `K=4` as the default pseudo-positive rollout profile and backward-compatible access to the legacy `K=2` trainer path.

## Requirements

### Validated

- ✓ Existing Stage-2 Channel-B uses a canonical `K=2` anchor-plus-one-explorer rollout contract with one edited anchor target — existing
- ✓ Existing Stage-2 teacher-forcing pipeline already supports matched-clean, FN injection, dead-anchor suppression, and plugin-owned coord-side losses — existing
- ✓ Existing OpenSpec change artifacts for `study-channel-b-pseudopositive-promotion` are drafted and validated in this worktree — existing

### Active

- [ ] Implement versionless `stage2_ab.channel_b.pseudo_positive.*` config with strict schema validation, a default pseudo-positive `K=4` profile, and legacy `K=2` compatibility.
- [ ] Extend Channel-B rollout preparation from one explorer to arbitrary `K-1` explorers with deterministic aggregate observability and fixed-denominator failure semantics.
- [ ] Implement support-count / support-rate pseudo-positive triage, overlap clustering, and anchor-owned coord targets exactly as specified by the OpenSpec change.
- [ ] Wire pseudo-positive objects into coord-only loss application, preserve one-forward training, and keep dead-anchor suppression narrow and duplicate-like only.
- [ ] Preserve backward-compatible triage and rollout metrics while adding the new pseudo-positive metadata, rates, and `best-K`-ready observability.
- [ ] Verify the change with config-contract tests, Stage-2 trainer tests, and a smoke-ready validation path for arbitrary-`K` ablations.

### Out of Scope

- Pseudo-positive desc CE or matched-prefix structure CE — explicitly deferred by the authored spec.
- Semantic-desc gating for pseudo-positive selection — deferred until the geometry-first slice is stable.
- Full-object negative CE for dead anchors — excluded to preserve the conservative supervision contract.
- Broad trainer redesign beyond the exact OpenSpec implementation surface — avoided to keep implementation bounded and reviewable.

## Context

This work happens inside the brownfield CoordExp repository, but the planning scope is intentionally narrow: implement the finalized OpenSpec change in `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/`. The most important grounded artifacts are the change `proposal.md`, `tasks.md`, and the delta specs for `stage2-ab-training`, `teacher-forcing-unified-loss-registry`, and `trainer-metrics-components`.

The implementation authority is concentrated in the current Stage-2 surfaces:
- `src/config/schema.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/target_builder.py`
- `src/trainers/stage2_two_channel/rollout_views.py`
- `src/trainers/teacher_forcing/modules/*`

This project should treat the authored OpenSpec as the source of truth, use the existing repo docs precedence from `docs/PROJECT_CONTEXT.md` → `docs/SYSTEM_OVERVIEW.md` → `docs/IMPLEMENTATION_MAP.md`, and preserve the current one-forward Stage-2 training architecture.

## Constraints

- **Governance**: OpenSpec-first implementation only — the change must match the authored delta specs exactly.
- **Compatibility**: Preserve legacy `K=2` behavior when `pseudo_positive.enabled=false`, and keep enabled `K=2` available as an explicit no-promotion control — existing configs and tests must remain valid.
- **Architecture**: Keep one edited-anchor teacher-forced forward — no second teacher trajectory or union ordering.
- **Observability**: Arbitrary-`K` metrics must stay deterministic and backward-compatible — `rollout/explorer/*`, `train/triage/unlabeled_consistent_count`, and recovered-GT rates need explicit semantics.
- **Validation**: Use the repo’s existing Python test path via `conda run -n ms` — implementation is not complete until targeted tests and smoke validation are planned.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Implement the exact `study-channel-b-pseudopositive-promotion` change, not a broader Stage-2 redesign | Keeps scope bounded and reviewable | — Pending |
| Use `K=4` as the default pseudo-positive profile and retain legacy `K=2` as the compatibility / control setting | Keeps the implementation target concrete without losing backward compatibility or ablation controls | — Pending |
| Use `support_count >= 2` plus `support_rate >= 2/3` for promotion | Prevents single-hit promotion and keeps enabled `K=2` as a no-promotion control | — Pending |
| Keep pseudo-positive supervision coord-only with anchor-owned target geometry | Matches the finalized OpenSpec and minimizes risky text-side supervision | — Pending |
| Preserve legacy explorer metrics as mean-over-valid-explorer-view summaries | Makes arbitrary-`K` observability interpretable without breaking current surfaces | — Pending |

---
*Last updated: 2026-03-22 after GSD initialization*

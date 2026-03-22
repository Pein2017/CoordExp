# Feature Research

**Domain:** Brownfield Stage-2 training feature implementation for pseudo-positive Channel-B
**Researched:** 2026-03-22
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Versionless typed `stage2_ab.channel_b.pseudo_positive.*` config | The change is unusable without a stable YAML surface and strict schema validation | MEDIUM | Must preserve disabled-path compatibility and reject unknown/versioned keys. |
| Arbitrary-`K` rollout preparation with `1` anchor + `K-1` explorers | This is the core mechanical capability behind the authored change | HIGH | Must keep deterministic explorer identity and one-forward training semantics. |
| Support-count / support-rate triage for unmatched anchor objects | This is the actual pseudo-positive selection mechanism | HIGH | Must include anchor unmatched gate, explorer unmatched gate, GT-conflict exclusion, and the `support_count >= 2` floor. |
| Overlap clustering and winner selection for pseudo-positive anchors | Prevents duplicate-like unmatched anchors from all being promoted | MEDIUM | Must be deterministic: connected components, highest support rate, then anchor order. |
| Coord-only pseudo-positive loss wiring | This is the intended supervision change | MEDIUM | Must affect `bbox_geo`, `coord_reg`, and `bbox_size_aux` only, not text-side CE. |
| Narrow duplicate-like dead-anchor suppression | Prevents dead branches from disappearing without preserving all-dead negative CE | MEDIUM | Must keep dead anchors out of the target and suppress only duplicate-like first-divergence branches. |
| Arbitrary-`K`-safe observability and compatibility metrics | Operators need comparable `best-K` ablations and existing dashboards/tests must stay interpretable | HIGH | Includes legacy explorer metric aggregation, shielded count compatibility, and failure telemetry semantics. |
| YAML profile and smoke-ready validation path | The feature is not operational until it can be authored, loaded, and exercised in a realistic run | MEDIUM | Needs one explicit config profile plus targeted tests and smoke checks. |

### Differentiators (Competitive Advantage / Later Extensions)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Broader `best-K` ablation suite beyond the first two-point study | Makes the new rate-based triage directly useful for research iteration | MEDIUM | Enabled by the arbitrary-`K` contract, but not required to land the first implementation slice. |
| Richer pseudo-positive weighting schedules by support rate | Could improve training stability once the basic contract is proven | MEDIUM | Deferred because v1 intentionally keeps one `coord_weight`. |
| Semantic-desc gating or semantic vote refinement | Could improve pseudo-positive precision in ambiguous scenes | HIGH | Explicitly out of v1 scope and should not block the geometry-first implementation. |
| Explorer-specific diagnostics beyond aggregate compatibility summaries | Useful for later deep debugging and ablations | MEDIUM | Deferred except for indexed carriers such as `dead_explorer_indices_by_view`. |

### Anti-Features (Commonly Tempting, Wrong For v1)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Pseudo-positive desc CE | It seems like a natural way to “teach” the model the object | It would turn uncertain unmatched objects into full text-supervised positives and violate the authored v1 contract | Keep pseudo-positive supervision coord-only. |
| Full-object negative CE for all dead anchors | It looks like a simple way to discourage bad objects | In incomplete-GT scenes it over-penalizes potentially real but unsupported objects and widens the negative contract too far | Keep duplicate-like first-divergence suppression only. |
| Explorer-only pseudo-positive objects | It appears to unlock more hidden recall | It changes the anchor-centric target contract and risks a much larger redesign | Stay anchor-centric in v1 and revisit only after stable evidence. |
| New CLI flags for pseudo-positive modes | Seems convenient for experimentation | Breaks the repo’s config-first contract and creates an undocumented parallel surface | Keep all controls in YAML under `stage2_ab.channel_b.*`. |
| Silent denominator shrinkage when explorers fail | Seems resilient operationally | Breaks reproducibility and makes support rates incomparable across runs | Hard-abort explorer-prep failure and report it via failure telemetry. |

## Feature Dependencies

```text
Config contract
    └──requires──> arbitrary-K rollout scheduling
                          └──requires──> support counting / rate computation
                                                  └──requires──> pseudo-positive promotion and loss wiring

Observability / compatibility
    └──requires──> arbitrary-K rollout scheduling
    └──requires──> support counting / rate computation

Dead-anchor duplicate-like suppression
    └──depends on──> final kept-anchor target boundaries
    └──depends on──> triage bucket assignment

YAML profile + smoke validation
    └──requires──> all prior implementation slices
```

### Dependency Notes

- **Config contract requires rollout scheduling:** the trainer cannot legally enter arbitrary-`K` mode until the YAML/schema surface is typed and validated.
- **Rollout scheduling requires support computation:** explorer evidence is only useful if it is carried through to triage as support counts/rates.
- **Support computation requires loss wiring:** pseudo-positive selection is only observable in training once selected anchors participate in coord-side loss groups.
- **Observability depends on the runtime and triage layers:** aggregate metrics and per-sample metadata are derived from the same rollout and triage state.
- **Dead-anchor suppression depends on final target shape:** duplicate-like suppression targets are defined relative to the final edited clean target, not the raw rollout.

## MVP Definition

### Launch With (v1)

- [ ] Typed `pseudo_positive` config with strict invariants and disabled-path compatibility
- [ ] Arbitrary-`K` rollout preparation with deterministic explorer identity and fixed-denominator semantics
- [ ] Support-rate-based pseudo-positive triage with anchor-owned coord targets
- [ ] Coord-only pseudo-positive supervision and duplicate-like dead-anchor suppression under one forward
- [ ] Backward-compatible metrics plus new pseudo-positive observability
- [ ] YAML profile, targeted tests, and smoke validation for at least two enabled `K` values

### Add After Validation (v1.x)

- [ ] Broader `best-K` ablations beyond the initial two-point comparison — once the first enabled path is stable
- [ ] More granular monitoring payloads or richer explorer-local debug artifacts — once operators need deeper diagnosis

### Future Consideration (v2+)

- [ ] Semantic-desc gating or vote weighting — defer until geometry-first pseudo-positive promotion is stable
- [ ] Explorer-only pseudo-positive proposals — defer until the anchor-centric contract has strong evidence
- [ ] Dynamic pseudo-positive weighting schedules — defer until one-weight v1 has stable metrics and inference behavior

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Typed pseudo-positive config contract | HIGH | MEDIUM | P1 |
| Arbitrary-`K` rollout scheduling | HIGH | HIGH | P1 |
| Support-rate triage and recovered-GT integration | HIGH | HIGH | P1 |
| Pseudo-positive overlap clustering | HIGH | MEDIUM | P1 |
| Coord-only loss projection | HIGH | MEDIUM | P1 |
| Duplicate-like dead-anchor filtering | HIGH | MEDIUM | P1 |
| Compatibility metrics and docs | HIGH | MEDIUM | P1 |
| Extended ablation suite | MEDIUM | MEDIUM | P2 |
| Semantic gating | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for OpenSpec-complete implementation
- P2: Should have after the first implementation slice lands
- P3: Future research, explicitly deferred

## Competitor / Baseline Analysis

| Feature | Current Repo Baseline | OpenSpec v1 Change | Our Approach |
|---------|-----------------------|--------------------|--------------|
| Rollout evidence | Canonical `K=2` anchor + one explorer | Arbitrary `K`, opt-in only | Preserve `K=2` default and generalize explorer evidence when enabled. |
| Unmatched anchor handling | `shielded_anchor` is neutral context only | Some trusted unmatched anchors become coord-only pseudo-positives | Promote only anchors meeting support-count and support-rate gates. |
| Dead-anchor negative signal | Duplicate-like first-divergence suppression exists | Keep it narrow and boundary-local | Reuse the current mechanism; only filter eligible dead anchors more explicitly. |
| Metrics | Single-explorer-shaped observability | `best-K`-ready rates with compatibility preserved | Extend metrics without breaking legacy semantics. |

## Sources

- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/.planning/PROJECT.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/proposal.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/design.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/tasks.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/specs/channel-b-lightweight-pseudopositive-v1/spec.md`

---
*Feature research for: CoordExp Stage-2 pseudo-positive implementation*
*Researched: 2026-03-22*

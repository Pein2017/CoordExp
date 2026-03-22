# Phase 1: Config And Rollout Foundation - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the enabling infrastructure for the OpenSpec change: a typed pseudo-positive config surface, arbitrary-`K` Channel-B rollout scheduling, deterministic explorer identity, and explicit anchor-drop / explorer-abort failure semantics. It does not yet implement support-rate triage, pseudo-positive loss wiring, or metrics/doc expansion beyond what is strictly required to make the runtime contract legal and stable.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
- All implementation choices in this phase are at Claude's discretion because this is a pure infrastructure phase.
- Preserve the existing disabled-path legacy `K=2` behavior while making the default pseudo-positive path `K=4`.
- Keep all new controls YAML-first under `stage2_ab.channel_b.*`; do not add CLI flags or parallel config surfaces.
- Favor strict schema/runtime failures over silent fallback when pseudo-positive mode is enabled.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/config/schema.py` already owns strict Channel-B key validation through `Stage2ABChannelBTriagePosteriorConfig`, `Stage2ABChannelBConfig`, and `Stage2ABConfig`.
- `src/trainers/stage2_two_channel.py` already owns anchor/explorer rollout generation, per-step metrics, and fallback/error routing for Channel-B.
- `tests/test_stage2_ab_config_contract.py` and `tests/test_stage2_ab_training.py` already provide the narrow regression harness for this phase.

### Established Patterns
- Config contracts fail fast on unknown keys and deprecated aliases.
- Channel-B builds one edited anchor target and one teacher-forced forward; rollout generation is evidence gathering, not a second supervision path.
- Existing Stage-2 code keeps operator behavior YAML-first and preserves compatibility through explicit tests rather than loose fallbacks.

### Integration Points
- `src/config/schema.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/rollout_views.py`
- `tests/test_stage2_ab_config_contract.py`
- `tests/test_stage2_ab_training.py`

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None

</deferred>

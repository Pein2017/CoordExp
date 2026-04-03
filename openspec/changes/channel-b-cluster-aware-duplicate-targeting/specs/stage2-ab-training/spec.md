## MODIFIED Requirements

### Requirement: Stage-2 AB Channel-B uses anchor-rooted rollout triage with legacy K=2 compatibility and default K=4 pseudo-positive evidence
When `custom.trainer_variant: stage2_two_channel`, the canonical Channel-B
contract SHALL build its clean teacher-forced target from rollout evidence
rooted in the anchor clean sequence.

Normative behavior:
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, the canonical
  Channel-B contract uses exactly two rollout views:
  - one anchor rollout using greedy / deterministic decoding,
  - one explorer rollout using stochastic decoding configured under
    `stage2_ab.channel_b.triage_posterior`,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, the opt-in
  pseudo-positive contract uses exactly
  `stage2_ab.channel_b.triage_posterior.num_rollouts` total rollout views:
  - one anchor rollout using greedy / deterministic decoding,
  - `num_rollouts - 1` explorer rollouts using the shared stochastic decode
    profile configured under `stage2_ab.channel_b.triage_posterior`,
  - repo-authored default pseudo-positive profiles SHOULD set `num_rollouts`
    to `4`,
- each rollout MUST independently reuse the existing bounded salvage + strict
  record acceptance + bbox-valid filtering + duplicate-like grouping +
  Hungarian matching path,
- duplicate-like grouping MUST be deterministic and MUST operate on parsed
  bbox objects before triage,
- duplicate-like grouping MUST be able to merge local same-description repeats
  that are not strictly sequential neighbors in emission order,
- GT-backed semantics MUST inherit the existing Channel-B accepted-clean
  Hungarian + gating contract,
- the final positive target MUST be built by editing the anchor clean sequence
  rather than rebuilding a union order,
- explorer-only non-GT-backed objects MUST be treated as dead by default,
- a GT hit found only on one or more explorer views MUST project to
  `recovered_fn`, not to anchor retention,
- duplicate-like objects that are spatially spread or explorer-supported MUST
  be eligible for crowd-safe exemption rather than automatic duplicate
  suppression,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true` and the anchor view
  fails to complete accepted-clean preparation, the sample MUST be dropped from
  Channel-B training for that step,
- the enabled pseudo-positive contract MUST NOT use the canonical empty-prefix
  fallback for malformed anchor preparation.

#### Scenario: Enabled pseudo-positive drops malformed anchor samples
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** the anchor rollout does not complete accepted-clean preparation for a
  sample
- **THEN** that sample is dropped from Channel-B training for the step
- **AND** the trainer does not fall back to the empty-prefix FN-only path for
  that sample.

#### Scenario: Local same-description duplicate cluster is collapsed before triage
- **WHEN** an anchor rollout emits multiple same-description bbox objects that
  form one deterministic local duplicate-like cluster
- **THEN** the Channel-B preparation keeps one deterministic survivor on the
  clean anchor surface
- **AND** the remaining members are carried as duplicate-candidate
  continuations rather than additional kept anchor objects.

#### Scenario: Spread-out same-description crowded objects avoid duplicate collapse targeting
- **WHEN** an anchor rollout emits multiple same-description bbox objects that
  are spatially separated enough to satisfy the crowd-safety rule
- **THEN** Channel-B preparation does not force those objects into one
  duplicate-suppression cluster
- **AND** the rollout can retain multiple kept objects subject to the existing
  matching and triage rules.

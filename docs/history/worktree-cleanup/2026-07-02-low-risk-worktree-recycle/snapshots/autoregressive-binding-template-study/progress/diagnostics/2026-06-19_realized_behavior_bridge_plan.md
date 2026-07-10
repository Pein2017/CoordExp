---
doc_id: progress.diagnostics.realized_behavior_bridge_plan
layer: progress
doc_type: diagnostic-plan
status: branch-provenance
domain: research-history
summary: Resolved Phase 4 plan for the checkpoint-928 binding-template realized before/after behavior bridge, including guidance separability and read-only object pointer trajectory tomography.
tags: [progress, diagnostics, autoregressive-binding, realized-behavior-bridge, phase4]
updated: 2026-06-19
branch: codex/autoregressive-binding-template-study
---

# Realized Behavior Bridge Plan

## Decision

Continue in the existing worktree:

```text
/data/CoordExp/.worktrees/autoregressive-binding-template-study
```

Use the current branch:

```text
codex/autoregressive-binding-template-study
```

Do not create a new worktree for this round. The Phase 3 analysis helpers,
fixed artifact contracts, and convergence note are the correct base for Phase 4.

Phase 4 is named:

```text
realized_behavior_bridge
```

The canonical missing artifact is:

```text
realized_before_after_behavior_rows.jsonl
```

## Rationale

The Phase 3 convergence pass demoted the strong raw duplicate-probability and
previous-anchor causal stories. What survived was a narrower observational
rank/identity shape, but all behavioral patch cases were still candidate-only.

The next high-value bridge is therefore not latent patching or micro-training.
It is a deterministic short-continuation bridge that asks whether candidate-only
identity/rank evidence corresponds to real next-object behavior under
prefix-side guidance and strict controls.

## Consequence

The next implementation plan lives at:

```text
docs/superpowers/plans/2026-06-19-realized-behavior-bridge.md
```

Primary output root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge
```

Required outputs:

```text
bridge_case_manifest.json
bridge_case_manifest.md
rendered_prefix_interventions.jsonl
realized_before_after_behavior_rows.jsonl
realized_before_after_behavior_summary.json
control_outcome_rows.jsonl
guidance_separability_matrix.json
guidance_separability_matrix.md
object_pointer_trajectory_rows.jsonl
object_pointer_trajectory_summary.json
object_pointer_trajectory_summary.md
promotion_gate_summary.json
run_manifest.json
```

Stage boundary:

```text
Stage A: realized_behavior_bridge
  allowed: deterministic short continuation, prefix/language/coordinate guidance, controls, guidance separability matrix, read-only object pointer trajectory tomography
  forbidden: hidden-state patching, attention patching, model weight updates, micro-training

Stage B: latent_behavior_patch
  allowed only after Stage A shows behavior movement beating controls

Stage C: bounded_micro_training_probe
  allowed only after a later findings note promotes it
```

## Evidence

- Scope: `none-yet`
- Current evidence source:
  - `progress/diagnostics/2026-06-18_binding_mechanism_phase3_audit_adjusted_findings.md`
  - `progress/diagnostics/2026-06-18_binding_mechanism_recursive_convergence.md`
- Phase 3 artifact root:
  - `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted`
- Phase 4 plan:
  - `docs/superpowers/plans/2026-06-19-realized-behavior-bridge.md`

## Resolved Grill Defaults

- Reuse current worktree and current branch.
- Build a new stratified case manifest rather than only the four
  `behavioral_patch_plan.json` candidate-only cases.
- Use family-specific events:
  - `desc_first`: `pre_desc`, `desc_end`
  - `geometry_first`: `pre_box_start`, `pre_x1`
- Treat `before` as original deterministic short continuation.
- Treat `after` as prefix-side guidance counterfactual continuation.
- Include controls in `control_outcome_rows.jsonl`.
- Include `guidance_separability_matrix.json` and
  `guidance_separability_matrix.md` as first-class bridge outputs, classifying
  whether behavior moves under desc-only, coord-only, joint desc+coord,
  object-start, no-op, or contaminated control guidance.
- Include `object_pointer_trajectory_rows.jsonl`,
  `object_pointer_trajectory_summary.json`, and
  `object_pointer_trajectory_summary.md` as read-only tomography outputs over
  semantic, spatial, coverage, and termination pointer hypotheses. These rows
  may consume logits, candidate mass, role rows, coord rows, or read-only
  hidden-state captures, but must not alter activations.
- Promote to latent patching only if target-guided movement beats controls,
  parse is mostly preserved, the effect appears outside discovery, guidance
  separability is not control-contaminated, and the object pointer trajectory
  does not contradict the claimed guidance channel.
- Do not invoke the existing micro-training planner in Phase 4.

## Open Follow-Up

Implement the plan, then write:

```text
progress/diagnostics/2026-06-19_realized_behavior_bridge_findings.md
```

The findings note must report whether the bridge is still `none-yet`, `smoke`,
or `scaled`, and whether the promotion gate recommends latent behavior patching,
a different hypothesis family, or a bridge redesign.

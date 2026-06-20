---
doc_id: progress.diagnostics.binding_mechanism_recursive_convergence
layer: progress
doc_type: diagnostic-convergence-report
status: branch-provenance
domain: research-history
summary: Recursive convergence pass for the checkpoint-928 binding-template Phase 3 evidence bundle, preserving the narrowed observational mechanism and the missing behavioral bridge.
tags: [progress, diagnostics, autoregressive-binding, recursive-convergence, phase3, behavioral-bridge]
updated: 2026-06-18
branch: codex/autoregressive-binding-template-study
---

# Binding Mechanism Recursive Convergence

## Scope

This note records the Task 12 recursive convergence pass for the checkpoint-928
`desc_first` versus `geometry_first` `compact_object_box_closed` binding-template
study.

Primary artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted
```

Inputs:

```text
progress/diagnostics/2026-06-18_binding_mechanism_phase3_audit_adjusted_findings.md
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/repetition_penalty_gate_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/null_leaderboard.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/identity_posterior_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/behavioral_patch/behavioral_patch_plan.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/hypothesis_registry.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/next_research_actions.json
```

The research-loop synthesis was run with the behavioral patch plan as the gate
summary source:

```bash
PHASE3=/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted
python scripts/analysis/run_autoregressive_binding_research_loop.py \
  --hypothesis-json "$PHASE3/hypothesis_registry.json" \
  --gate-summary-json "$PHASE3/behavioral_patch/behavioral_patch_plan.json" \
  --output-root "$PHASE3"
```

I used `behavioral_patch/behavioral_patch_plan.json` rather than
`null_leaderboard.json` because it preserves the explicit behavioral gate
metadata for the final convergence pass: `gate_passes: false`, the required gate
names, selected candidate-only cases, controls, and `short_rollout_ready: false`.
The null leaderboard remains direct evidence for held-out/null failure, but it
does not carry the behavioral-patch readiness boundary by itself.

Canonical outputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/hypothesis_registry.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/next_research_actions.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/next_research_actions.md
```

The pre-existing scratch directory
`/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/research_loop_probe_tmp`
was not used as a canonical output and was left untouched.

## Expanded Mandate

The recursive pass was asked to decide whether the Phase 3 family has converged,
failed, or still needs a bridge probe. The mandate is wider than the initial
duplicate-probability story:

- preserve audit-adjusted gates over repetition penalty, anchor circularity,
  held-out/null structure, and behavioral-readiness metadata;
- separate candidate-only evidence from realized next-object behavior;
- avoid claiming a deepest mechanism from teacher-forced or candidate-ledger
  artifacts;
- name the next recursive branch that can connect candidate identity or rank
  fragility to actual behavior movement.

## Evidence Graph

Repetition-penalty gate:

- Rows: `2098`.
- Duplicate versus new raw `p_emitted` gap was `+0.0162`, but the
  penalty-adjusted gap flipped to `-0.0266`.
- Duplicate versus non-duplicate raw `p_emitted` gap was `+0.0206`, but the
  penalty-adjusted gap flipped to `-0.0232`.
- Rank gaps survived directionally but weakened:
  duplicate-new `+0.8160` raw rank gap to `+0.3246` penalty-adjusted, and
  duplicate-non-duplicate `+0.6799` to `+0.2531`.

Held-out/null leaderboard:

- Joined rows: `6928`.
- Split counts: `2756` discovery, `1344` reserve, `2828` val200 remainder.
- Family counts: `3324` `desc_first`, `3604` `geometry_first`.
- For `next_step_duplicate_onset`, simple `object_idx` baselines were among the
  strongest reserve and val200-remainder signals. Candidate-mass signals did not
  clearly beat null/phase structure.
- For `next_step_unmatched_onset`, held-out accuracy was also dominated by
  simple signals and class imbalance rather than a clean mechanism-specific
  candidate mass result.

Same-class identity posterior:

- Rows: `6480`, candidate-only and matched-target-only.
- Penalty-adjusted target identity margins were positive across both families
  and all splits:

| family | split | penalty-adjusted target margin | target identity mass |
| --- | --- | ---: | ---: |
| `desc_first` | `discovery` | 0.3258 | 0.4448 |
| `desc_first` | `reserve` | 0.4660 | 0.5519 |
| `desc_first` | `val200_remainder` | 0.4687 | 0.5150 |
| `geometry_first` | `discovery` | 0.2493 | 0.3804 |
| `geometry_first` | `reserve` | 0.3697 | 0.4565 |
| `geometry_first` | `val200_remainder` | 0.4355 | 0.4813 |

Behavioral patch plan:

- Stage: `plan`.
- Cases selected: `4`, all `candidate_only`.
- Controls listed: `9`.
- `gate_passes: false`.
- `short_rollout_ready: false` for every selected case.
- Block reason: candidate-only patch cases have no realized before/after
  behavior rows.

Research-loop synthesis:

- Records: `7`.
- Actions: `7`.
- Every hypothesis family remained in observation mode because required gates
  were missing rather than passed.
- Normalized behavioral gates reported `gate_passes: false` and
  `short_rollout_ready: false`.

## Hypotheses Promoted

No hypothesis is promoted to causal convergence.

Two narrower observational shapes are worth carrying forward:

- Rank/selection fragility survives repetition-penalty correction, even though
  the stronger raw duplicate-probability story does not.
- Candidate identity margins exist in matched-target rows for both template
  families and all splits, especially after penalty adjustment.

These are promoted only as bridge-probe candidates, not as realized behavioral
mechanisms.

## Hypotheses Demoted

The strong raw duplicate-probability hypothesis is demoted. Repetition-penalty
correction flips the duplicate `p_emitted` advantage against both new and
non-duplicate rows.

The strong previous-anchor causal story is also demoted. Anchor-distance and
candidate-field observations remain useful, but the held-out/null leaderboard
does not show mechanism-specific signals cleanly beating simple phase, index, or
count structure, and no behavioral movement has been produced.

The behavioral-causal claim is not supported by this bundle. The patch artifact
is a plan over candidate-only rows, not a realized before/after intervention.

## Hypotheses Still Alive

The following families remain live but unresolved:

- `H-decoder-artifact`: repetition penalty or decode processors may shape the
  apparent fragility; rank survives where raw probability does not.
- `H-coverage-ledger`: emitted-vs-remaining state may exist, but candidate mass
  and held-out prediction have not crossed the causal bridge.
- `H-identity-binding`: same-class target margins are visible in matched-target
  rows, but the evidence excludes unmatched/no-target groups and has no
  behavior movement.
- `H-visual-blindness-vs-guidance`: false-negative recoverability remains a
  possible contributor outside this candidate-only posterior.
- `H-coordinate-basin`: coordinate-token basin attraction remains plausible from
  broader diagnostic history, but this pass did not isolate it causally.
- `H-termination-basin`: stop or parse competition may explain some
  next-object failures, especially where no candidate-only bridge rows exist.

## Most Plausible Current Mechanism

The most plausible current mechanism is a shallow selection-fragility account:
the model exposes candidate identity and rank structure around next-object
selection, but this structure is entangled with decode penalty, object position,
family-specific timing, and class/count baselines.

This is not yet a deepest mechanism. The evidence says the model can place mass
on matched target identity buckets and can show residual rank separation after
penalty adjustment, but the current artifacts do not prove that this state
causes the next emitted object under free continuation.

## Alternative Explanations Still Capable Of Explaining The Evidence

Decoder and repetition-penalty artifacts can explain why raw probability gaps
looked stronger than the penalty-adjusted result.

Object index, phase, same-class count, and split imbalance can explain much of
the held-out/null prediction surface without requiring a binding mechanism.

Candidate-ledger construction can make target margins visible in matched-target
rows while saying little about unmatched or no-target rows.

Coverage-state, coordinate-basin, visual-guidance, and termination-basin
accounts can still explain subsets of the observed failures. This pass does not
falsify them because it did not run realized continuation, causal patching, or a
model perturbation, and because candidate-only rows do not supply realized
before/after bridge evidence by themselves.

## Recommended Recursive Branch

Build the missing bridge artifact before any scaled causal claim:

```text
realized_before_after_behavior_rows.jsonl
```

The branch should:

- derive explicit boolean gate metadata from the Phase 3 summaries;
- select candidate-only cases with matched targets and clear controls;
- materialize before/after behavior rows before invoking short rollout;
- include noop, wrong-role, wrong-image, post-commit, same-image wrong-object,
  shuffled-label, norm-matched random, and repetition-penalty on/off controls;
- report any movement as next-object behavior, not candidate mass.

If this bridge produces behavior movement, then the next recursive step can be
causal patching over selected cases. If it does not, the current family should be
redirected toward the surviving alternative families rather than training.

## Micro-Training Dry Runs

Not run.

The generated `next_research_actions.json` does not select a training
perturbation as the next justified branch. It keeps all seven families in
observation mode with missing behavioral gates, and the behavioral patch plan
marks `short_rollout_ready: false`. A dry-run training manifest would be
premature before the realized before/after behavioral bridge exists.

## GPU Artifacts

No new GPU run was launched for Task 12.

This note reuses the existing Phase 3 replay evidence summarized in
`2026-06-18_binding_mechanism_phase3_audit_adjusted_findings.md`: the scaled
probe replay saw `8` shards, no shard errors, `75528` attention rows, `8392`
coord-logit rows, and `226584` role rows.

The Task 12 research-loop synthesis itself is a CPU/metadata synthesis over JSON
artifacts and wrote only the canonical research-loop outputs listed above.

## Convergence Status

Convergence status: `not_converged_continue`.

The current evidence narrows the family but does not converge to a causal or
deep mechanism. The strong raw duplicate-probability and previous-anchor causal
versions are demoted. The surviving evidence is an observational rank/selection
and matched-target identity shape. The next recursive branch should materialize
realized before/after behavioral rows with explicit gate metadata and controls.

---
title: Prefix-State Transition Tomography Phase A3.1 Analysis
date: 2026-06-04
status: completed-diagnostic
owner: codex
depends_on:
  - progress/diagnostics/2026-06-04_prefix_state_transition_phase_a3_1_launch.md
  - outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096/phase_a3_analysis/phase_a3_analysis_summary.json
  - outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096/phase_a3_analysis/phase_a3_analysis_report.md
---

# Prefix-State Transition Tomography Phase A3.1 Analysis

This note records completed analysis results for Phase A3.1.  Evidence scope is
limited to paired checkpoint-3664 prefix-state tomography.  The run compares
ET-RMP-CE and pure-CE readouts on sampled train/val prefix states.  It is not a
full validation metric and does not include attention-head causal intervention.

## Artifact Roots

Primary artifact root:

`outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096`

Analysis output root:

`outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096/phase_a3_analysis`

Primary analysis files:

- `phase_a3_analysis_summary.json`
- `phase_a3_analysis_report.md`
- `quadrant_by_role_transition.json`
- `forced_x1_by_role_transition.json`
- `paired_delta_by_transition.json`

Summary plot files:

- `plots/boundary_alignment_by_transition.png`
- `plots/quadrant_by_transition.png`
- `plots/forced_x1_coverage_and_peaks.png`
- `plots/paired_delta_by_prefix_depth.png`
- `plots/boundary_bad_x1_good_alignment_breakdown.png`

## Completion Evidence

Status check after completion:

- `stage_status`: `final_artifacts_present`
- `final_ready`: `true`
- `shards_complete`: `true`
- `alive_shard_processes`: `0`
- `expected_shards`: `8`
- `failed_launch_gates`: `[]`

Final row counts:

```json
{
  "prefix_state_sampled_rows": 4096,
  "boundary_score_rows": 26840,
  "boundary_decision_rows": 8192,
  "forced_x1_rows": 15138,
  "quadrant_rows": 15138
}
```

Sample composition:

- train rows: `2087`
- val rows: `2009`
- same-desc transition rows: `2872`
- different-desc transition rows: `1224`
- same-desc prefix rows: `2872`
- different-desc prefix rows: `816`
- class-block prefix rows: `408`

## Main Readout Definitions

`boundary_good` means the boundary/full-desc scoring stage favored a residual
object description over EOS, emitted objects, or ties.

`x1_good` means the forced-desc pre-x1 readout assigned at least one selected x1
peak to a residual same-desc GT x1.

`both_good` means both conditions were true for the same paired state.

`boundary_bad_x1_good` is the decoupling bucket where forced-desc x1 evidence
exists, but boundary continuation or desc selection does not favor the residual
object.

`boundary_good_x1_bad` is the complementary bucket where continuation/desc
selection is favorable, but forced x1 binding does not cover a residual
instance.

## Checkpoint-Level Results

Across all paired forced-x1 rows:

| checkpoint | rows | boundary_good | x1_good | both_good | boundary_bad_x1_good | boundary_good_x1_bad | both_bad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ET-RMP-CE | 7569 | 0.4110 | 0.5306 | 0.2443 | 0.2863 | 0.1667 | 0.3027 |
| pure-CE | 7569 | 0.5273 | 0.5496 | 0.3247 | 0.2249 | 0.2025 | 0.2479 |

Across unique boundary decisions:

| checkpoint | rows | residual_favored | eos_favored | emitted_favored | mixed_or_tied | mean residual-vs-EOS margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ET-RMP-CE | 4096 | 0.4214 | 0.2971 | 0.0369 | 0.2446 | 4.1834 |
| pure-CE | 4096 | 0.5488 | 0.0889 | 0.0623 | 0.3000 | 6.6467 |

Forced-desc x1 summary:

| checkpoint | rows | mean residual x1 coverage | coverage > 0 | coverage = 1 | mean merged peak count | median x1 target rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ET-RMP-CE | 7569 | 0.2765 | 0.5306 | 0.0898 | 2.8905 | 70 |
| pure-CE | 7569 | 0.3516 | 0.5496 | 0.1580 | 4.2792 | 73 |

## Same-Desc Vs Different-Desc

Same-desc transition is harder for both checkpoints than different-desc
transition.

| checkpoint / transition | rows | boundary_good | x1_good | both_good | both_bad |
| --- | ---: | ---: | ---: | ---: | ---: |
| ET-RMP-CE / different-desc | 2448 | 0.5678 | 0.6597 | 0.3766 | 0.1491 |
| ET-RMP-CE / same-desc | 5121 | 0.3361 | 0.4689 | 0.1810 | 0.3761 |
| pure-CE / different-desc | 2448 | 0.7255 | 0.6834 | 0.5098 | 0.1009 |
| pure-CE / same-desc | 5121 | 0.4325 | 0.4856 | 0.2363 | 0.3181 |

Boundary decisions show the same split:

| checkpoint / transition | rows | residual_favored | eos_favored | emitted_favored | mixed_or_tied | mean residual-vs-EOS margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ET-RMP-CE / different-desc | 1224 | 0.5678 | 0.3088 | 0.1234 | 0.0000 | 1.0270 |
| ET-RMP-CE / same-desc | 2872 | 0.3590 | 0.2921 | 0.0000 | 0.3489 | 5.5286 |
| pure-CE / different-desc | 1224 | 0.7255 | 0.0662 | 0.2083 | 0.0000 | 3.3939 |
| pure-CE / same-desc | 2872 | 0.4735 | 0.0985 | 0.0000 | 0.4279 | 8.0329 |

## Paired ET-RMP-CE Vs Pure-CE Deltas

Across all paired rows:

- mean `pure_minus_et` forced-x1 residual coverage: `+0.0751`
- pure coverage greater than ET rate: `0.2574`
- ET coverage greater than pure rate: `0.0753`
- coverage tie rate: `0.6673`
- pure-only boundary-good rate: `0.1527`
- ET-only boundary-good rate: `0.0365`
- quadrant disagreement rate: `0.2453`

By transition type:

| transition | paired rows | mean pure-minus-ET coverage | pure coverage > ET | ET coverage > pure | pure-only boundary-good | ET-only boundary-good | quadrant disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| different-desc | 2448 | 0.0793 | 0.2896 | 0.0850 | 0.2002 | 0.0425 | 0.3092 |
| same-desc | 5121 | 0.0731 | 0.2419 | 0.0707 | 0.1301 | 0.0336 | 0.2148 |

By prefix depth:

| prefix depth | paired rows | mean pure-minus-ET coverage | pure coverage > ET | ET coverage > pure | quadrant disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| empty | 1798 | 0.1011 | 0.2686 | 0.0306 | 0.1340 |
| shallow_1 | 2490 | 0.0954 | 0.3269 | 0.0863 | 0.3129 |
| mid_half | 2465 | 0.0423 | 0.2057 | 0.1124 | 0.2592 |
| class_block_done | 816 | 0.0550 | 0.1765 | 0.0282 | 0.2426 |

## Findings

1. Same-desc state transition is a stronger bottleneck than different-desc
   transition in this readout.  Both checkpoints have lower `boundary_good`,
   lower `x1_good`, lower `both_good`, and higher `both_bad` on same-desc
   rows.

2. Pure-CE is less conservative at the boundary stage in this paired probe.
   It has higher residual-favored boundary rate and much lower EOS-favored
   rate than ET-RMP-CE.  The ET-RMP-CE EOS-favored rate is especially high in
   different-desc transitions.

3. Pure-CE exposes more forced-desc x1 candidate mass than ET-RMP-CE.  It has
   higher mean residual x1 coverage, higher full-coverage rate, and more merged
   x1 peaks.  This supports the observation that pure-CE often presents a
   broader candidate field at pre-x1.

4. ET-RMP-CE is not simply blind at x1.  Its `boundary_bad_x1_good` rate is
   substantial (`0.2863` overall, `0.2878` on same-desc prefix rows).  These
   rows have residual x1 evidence under forced desc, but the boundary decision
   does not favor the residual continuation.

5. The main failure is not a single uniform failure mode.  Both decoupling
   buckets are large: `boundary_bad_x1_good` and `boundary_good_x1_bad` both
   appear often.  This separates continuation/description selection failures
   from x1 binding failures.

6. Class-block prefix rows do not resolve the issue in this run.  They improve
   boundary rates relative to same-desc prefix rows for ET-RMP-CE, but x1
   coverage remains modest and pure-CE still has higher `both_good`.

7. The checkpoint difference is paired and robust within this 4096-row sample:
   pure-CE has greater forced-x1 residual coverage more often than ET-RMP-CE
   (`0.2574` vs `0.0753`), while most pairs tie (`0.6673`).

8. EOS is a major but incomplete explanation for boundary failures.  In the
   `boundary_bad_x1_good` bucket, ET-RMP-CE is EOS-favored in `0.5228` of rows
   overall, while pure-CE is EOS-favored in `0.1945`.  Same-desc rows are more
   often mixed/tied than purely EOS-favored: ET-RMP-CE same-desc
   `boundary_bad_x1_good` rows are `0.4267` EOS-favored and `0.5733`
   mixed/tied; pure-CE same-desc rows are `0.1621` EOS-favored and `0.8379`
   mixed/tied.

## Interpretation Boundaries

These results support a state-transition bottleneck framing for low recall:
the model can often expose residual x1 evidence when desc is forced, but it
does not consistently select the residual continuation at the autoregressive
boundary.

The results do not prove that attention clusters are the cause.  This run is
based on boundary scoring and forced pre-x1 posterior readouts, not attention
head ablation or patch-level causal intervention.

The results do not prove that pure-CE is better as a final detector.  This run
measures prefix-state readout behavior, not final rollout AP/recall.  Pure-CE
appears broader and less conservative in this probe; ET-RMP-CE may still have
better top-1 coordinate quality in other artifacts.

## Verification Commands

Analysis generation:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/analyze_phase_a3_results.py \
  --artifact-root /data/CoordExp/outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096
```

Status check:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/status.py \
  --artifact-root /data/CoordExp/outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096
```

Compile check:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -m py_compile /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/analyze_phase_a3_results.py
```

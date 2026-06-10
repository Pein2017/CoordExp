---
title: A3.2 Sorted-Random No-Newline Smoke Findings
date: 2026-06-04
status: diagnostic-evidence
owner: codex
depends_on:
  - outputs/analysis/autoreg_object_rollout/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2_smoke_retry2
  - docs/superpowers/specs/2026-06-04-post-x1-instance-basin-tomography-design.md
---

# A3.2 Sorted-Random No-Newline Smoke Findings

This note records the A3.2 smoke evidence that motivates A3.3 post-`x1`
instance-basin tomography.  The evidence scope is a mechanism smoke, not a full
validation benchmark and not a detector accuracy comparison.

## Scope

Artifact root:

```text
outputs/analysis/autoreg_object_rollout/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2_smoke_retry2
```

Checkpoint roles:

- `fullobj_random_pure_ce_ckpt3668`
- `fullobj_sorted_pure_ce_ckpt3668`

Decode policy:

```text
free_text_unconstrained_greedy_temp0
```

Template contract:

```text
compact_full_no_newline_native_v1
```

The rollout parser in this smoke is diagnostic and non-metric-bearing.  Official
detector metrics should come from strict evaluation artifacts, not from this
free-text mechanism parser.

## Key Counts

Prefix readout:

```json
{
  "merged_prefix_state_count": 64,
  "mean_sorted_minus_random_boundary_residual_favored_rate": 0.046875,
  "mean_sorted_minus_random_residual_vs_eos_margin": 1.0677893123279218,
  "mean_sorted_minus_random_strict_r95_x1_hit_rate": -0.0633716066919192
}
```

Boundary winner classes:

| checkpoint role | residual same-desc | residual other-desc | emitted other-desc | EOS |
| --- | ---: | ---: | ---: | ---: |
| `fullobj_random_pure_ce_ckpt3668` | 29 | 29 | 3 | 3 |
| `fullobj_sorted_pure_ce_ckpt3668` | 32 | 29 | 2 | 1 |

Strict x1 hits in prefix readout:

| checkpoint role | strict x1 hits | residual x1 checks | rate |
| --- | ---: | ---: | ---: |
| `fullobj_random_pure_ce_ckpt3668` | 107 | 418 | 0.2560 |
| `fullobj_sorted_pure_ce_ckpt3668` | 81 | 418 | 0.1938 |

FN probe:

```json
{
  "row_count": 768,
  "primary_bucket_counts": {
    "coord_binding_failure": 380,
    "desc_selection_failure": 224,
    "probe_invalid_or_unscored": 107,
    "rescued_residual_instance": 57
  }
}
```

Headline bucket ratio:

```text
coord_binding_failure=380/768
desc_selection_failure=224/768
rescued_residual_instance=57/768
```

FN prefix sensitivity:

```json
{
  "prefix_suppression_flip_count": 0,
  "prefix_condition_count_per_mode": 128
}
```

Slot evidence:

| slot | evidence source | strict hit | count |
| --- | --- | ---: | ---: |
| `x1` | `hint_control` | true | 366 |
| `x1` | `model_decode` | true | 43 |
| `x1` | `model_decode` | false | 252 |
| `y1` | `hint_control` | true | 186 |
| `y1` | `model_decode` | true | 124 |
| `y1` | `model_decode` | false | 351 |
| `x2` | `model_decode` | true | 159 |
| `x2` | `model_decode` | false | 502 |
| `y2` | `model_decode` | true | 200 |
| `y2` | `model_decode` | false | 461 |

## Findings

1. The smoke does not support a pure EOS-only explanation for low recall.
   Residual candidates are usually favored over EOS in the paired prefix
   readout, and sorted has a positive residual-vs-EOS margin delta over random.

2. The dominant FN probe bucket is coordinate / instance binding failure rather
   than only desc selection failure.

3. `x1` can be a useful control signal, especially when externally hinted, but
   post-`x1` closure into `y1/x2/y2` remains unstable in this smoke.

4. Sorted and random show different mechanism profiles.  Sorted is more
   continuation/residual-favored and has fewer desc-selection failures, while
   random exposes more strict x1 hits in this sampled prefix readout.

5. The smoke does not strongly support simple emitted-ledger suppression as the
   dominant cause.  The recorded `prefix_suppression_flip_count` is zero.

## Consequence

The next experiment should not repeat only FN rescue or final rollout scoring.
It should directly probe post-`x1` instance-basin dynamics:

- whether `desc+x1_i` anchors the model to instance `i`;
- whether `y1/x2/y2` drift to same-desc competitors or background;
- whether bad prefixes reshape the same target basin;
- whether the ET-RMP-CE reference checkpoint differs from the two no-newline
  pure-CE checkpoints in this basin geometry.

See:

```text
docs/superpowers/specs/2026-06-04-post-x1-instance-basin-tomography-design.md
```

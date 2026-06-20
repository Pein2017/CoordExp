---
doc_id: progress.diagnostics.binding_template_deep_probe_v1_findings
layer: progress
doc_type: diagnostic-findings
status: branch-provenance
domain: research-history
summary: First hidden-state, coord-logit, and attention replay findings for the checkpoint-928 desc-first versus geometry-first compact object/box-closed ablation.
tags: [progress, diagnostics, autoregressive-binding, field-order, hidden-states, attention, coord-logits]
updated: 2026-06-18
branch: codex/autoregressive-binding-template-study
---

# Binding Template Deep Probe V1 Findings

## Scope

This note records the first deterministic observational probe for the
checkpoint-928 `desc_first` versus `geometry_first` compact
`compact_object_box_closed` ablation.

Executable config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Probe artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_deep_probe_v1
```

Key files:

```text
case_surface_rows.jsonl
object_step_rows.jsonl
token_role_rows.jsonl
replay_role_rows.jsonl
coord_logit_rows.jsonl
attention_rows.jsonl
probe_report_summary.json
probe_report.md
```

Replay scope:

- COCO val first-200 operational panel, `bbox_len12000` canonical JSONL.
- 8 approved image IDs x 2 checkpoint families = 16 family-case surface rows.
- 14 parseable rendered replays; parse-failure cases stay in surface-token
  anatomy only.
- 8 GPU shards (`CUDA_VISIBLE_DEVICES=0..7`) completed with no shard errors.
- Merged replay rows:
  - hidden role rows: `14796`
  - coord-logit rows: `548`
  - attention rows: `4932`

Important loader caveat:

- `desc_first` checkpoint metadata saves a historical
  `coord_offset_adapter` module with `coord_ids` and `embed_offset`.
- `geometry_first` checkpoint metadata saves the newer
  `token_embeddings_adapter` module with `token_ids` and `embed_offset`.
- The probe loader installs a compatibility `coord_offset_adapter` module for
  `desc_first` and relocates the stale base path
  `/data/home/xiaoyan/AIteam/data/CoordExp/...` to the existing
  `/data/CoordExp/...` cache path when needed.
- This is recorded as a probe compatibility shim, not a silent normalization.

## Surface Anatomy

The surface stage wrote:

- `16` case rows
- `461` object-step rows
- `1233` token-role rows

Parse-failure split:

- `desc_first` image `1761`: strict-template mismatch, empty parsed prediction,
  but raw token stream exists.
- `geometry_first` image `3255`: strict-template mismatch, empty parsed
  prediction, but raw token stream exists.

Previous-pred duplicate counts using same-desc IoU >= `0.70`:

| family | duplicate objects | images |
| --- | ---: | --- |
| `desc_first` | 22 | `15254`, `2685`, `3255` |
| `geometry_first` | 48 | `1761`, `2685` |

Largest visible bursts:

- `desc_first` / `15254`: first previous-IoU duplicate at object `20`, with
  `16` repeated `broccoli` objects.
- `geometry_first` / `1761`: first previous-IoU duplicate at object `17`, with
  `25` repeated `person` objects.
- `geometry_first` / `2685`: first previous-IoU duplicate at object `3`, with
  `23` previous-IoU duplicates across `banana`, `bottle`, and `cup`.

This means GT-repeat matching alone is too narrow for burst onset: some
large bursts become a drift through locally valid but unmatched near-duplicate
anchors rather than a clean repeated matched-GT event.

## Coord-Token Basin Readout

The strict schema/type behavior is stable: coordinate-vocab mass is near 1.0
for all slots in both families. The instability is slot basin selection within
the coord-token subspace.

Mean emitted-rank by slot:

| family | x1 | y1 | x2 | y2 |
| --- | ---: | ---: | ---: | ---: |
| `desc_first` | 3.33 | 2.02 | 2.46 | 10.75 |
| `geometry_first` | 2.80 | 1.76 | 1.59 | 2.28 |

Mean coordinate-vocab mass at `x1`:

| family | coord-vocab mass |
| --- | ---: |
| `desc_first` | 0.99797 |
| `geometry_first` | 0.99733 |

The current evidence says `x1` is the first unstable basin/commitment slot,
not a schema/type failure. `desc_first` also shows a surprisingly weak `y2`
emitted-rank in this focused panel, which may reflect late-span drift after a
semantics-first branch has already selected a broad object trajectory.

For objects that are previous-pred duplicates under same-desc IoU >= `0.70`,
the `x1` emitted coordinate becomes less supported:

| family | previous-duplicate? | x1 n | mean rank | mean top1 distance | mean p(cond emitted) | mean norm entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `desc_first` | yes | 4 | 6.25 | 7.25 | 0.0175 | 0.785 |
| `desc_first` | no | 59 | 3.14 | 28.61 | 0.0475 | 0.722 |
| `geometry_first` | yes | 3 | 6.00 | 3.33 | 0.0150 | 0.842 |
| `geometry_first` | no | 71 | 2.66 | 66.49 | 0.0522 | 0.759 |

Interpretation: duplicate bursts are not simply “high confidence repeats.”
At the focused duplicate objects, the emitted `x1` is often lower probability
and higher entropy than non-duplicates. The model still emits the coordinate
inside the coordinate-token manifold, but the basin is fragile and locally
re-attracted.

## Attention Readout

Attention summaries are observational and averaged over selected late layers.
They are useful as a temporal readout, not as causal proof.

Repeated/unmatched objects attend more to previously emitted assistant content
before the current object is fully formed.

Examples:

| family | role | match kind | previous-object mass | current-object mass | wrapper mass |
| --- | --- | --- | ---: | ---: | ---: |
| `desc_first` | `pre_object` | new GT | 0.154 | 0.000 | 0.192 |
| `desc_first` | `pre_object` | repeated GT | 0.223 | 0.000 | 0.233 |
| `desc_first` | `pre_x1` | new GT | 0.028 | 0.140 | 0.167 |
| `desc_first` | `pre_x1` | repeated GT | 0.067 | 0.143 | 0.200 |
| `geometry_first` | `pre_object` | new GT | 0.132 | 0.000 | 0.238 |
| `geometry_first` | `pre_object` | repeated GT | 0.145 | 0.000 | 0.255 |
| `geometry_first` | `pre_x1` | new GT | 0.055 | 0.099 | 0.150 |
| `geometry_first` | `pre_x1` | repeated GT | 0.067 | 0.104 | 0.166 |

The useful signal is temporal: the previous-object and wrapper-token mass is
already higher at `pre_object` / `pre_x1`, before a human-visible duplicate is
complete. This supports a re-attraction / onset-dynamics lens more than a
post-hoc duplicate-count lens.

## Hidden-State Readout

Late hidden norms are higher for repeated or unmatched branches than for new
GT branches at the same decision roles. The effect is clearest in last-layer
groups at `pre_x1`, which is the same slot where coord-basin support weakens.

Examples:

| family | role | layer group | new GT norm | repeated GT norm | unmatched norm |
| --- | --- | --- | ---: | ---: | ---: |
| `desc_first` | `pre_x1` | `last` | 2276.5 | 2416.9 | 2534.2 |
| `geometry_first` | `pre_x1` | `last` | 2438.0 | 2536.6 | 2655.6 |
| `desc_first` | `pre_object` | `last` | 2003.8 | 2074.4 | 2104.1 |
| `geometry_first` | `pre_object` | `last` | 1936.7 | 1990.4 | 1998.0 |

This looks like a state-load or unresolved-branch signal rather than a clean
visual absence signal. The model is in a heavier, less locally stable state
when it enters repeated or unmatched emissions.

## First Mechanism Picture

The strongest current picture is:

1. The model stays inside the coordinate-token schema basin.
2. The first fragile binding step is not “should I output a coordinate token?”
   but “which coordinate basin anchors the next object?”
3. At the onset of repeated or unmatched branches, previous-prefix and wrapper
   attention are already elevated.
4. Late hidden norms increase at the same pre-object and pre-x1 boundaries,
   especially for repeated/unmatched objects.
5. The emitted x1 basin can be lower-confidence and higher-entropy for
   duplicates, so duplication is not merely an overconfident copy. It is closer
   to an autoregressive re-attraction path through locally valid schema tokens.

This is still observational. The next causal target should be the
`pre_object -> pre_x1` transition: patch or perturb previous-object/wrapper
state while holding the visual input and rendered prefix fixed, then measure
whether the selected x1 basin and next-object continuation move.

## False Negative / Guidance Implication

The low-recall examples do not currently look like simple visual blindness.
Several have parseable rendered replays, high coordinate-vocab mass, and
meaningful local coord distributions. The plausible failure is selection or
prefix guidance: valid objects can remain unselected while the autoregressive
state follows a repeated/unmatched branch.

Concrete next probe:

- For missing objects in the low-recall panel, render short language-side and
  coordinate-side guidance prefixes that name or spatially hint the missing
  object.
- Compare whether the missing object's coordinate basin appears under
  teacher-forced or short continuation contexts.
- Require manual review for ambiguous visual cases, but use data analysis to
  pre-rank likely guidance-rescuable false negatives.

## Boundaries

- Existing rollout artifacts use `temperature=0.0`, `rp=1.10`,
  `max_new_tokens=3084`.
- Surface anatomy is post-hoc over those artifacts.
- Replay logits/hidden/attention are teacher-forced on rendered parsed
  predictions, not free generation.
- Parse-failure cases contribute surface-token evidence but no rendered GPU
  replay in this version.
- Attention summaries are observational; no causal movement is claimed yet.
- The `coord_offset_adapter` compatibility shim must be kept visible in
  future interpretation because coord-token rows are central to the mechanism.

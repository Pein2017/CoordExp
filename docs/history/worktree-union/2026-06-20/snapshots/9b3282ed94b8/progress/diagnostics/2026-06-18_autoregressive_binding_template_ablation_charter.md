---
doc_id: progress.diagnostics.autoregressive_binding_template_ablation_charter
layer: progress
doc_type: diagnostic-charter
status: branch-provenance
domain: research-history
summary: Charter for the current-main desc-first versus geometry-first compact object/box-closed autoregressive binding study.
tags: [progress, diagnostics, autoregressive-binding, field-order, geometry-first, desc-first, compact-object-box-closed]
updated: 2026-06-18
branch: codex/autoregressive-binding-template-study
---

# Autoregressive Binding Template Ablation Charter

## Decision

Start a new current-main mechanistic study for the `desc_first` versus
`geometry_first` compact object/box-closed checkpoint pair. The study inherits
the previous mechanistic loop philosophy and reusable probe infrastructure, but
its evidence source of truth is this clean template ablation rather than the
older random-order/aux-loss continuation study.

The active worktree is:

```text
/data/CoordExp/.worktrees/autoregressive-binding-template-study
```

The branch is:

```text
codex/autoregressive-binding-template-study
```

## Checkpoints And Template Surface

The two checkpoint paths are intentionally asymmetric on disk and must be
recorded exactly in run manifests.

Geometry-first checkpoint:

```text
/data/CoordExp/output/stage1_2b/coco_bbox_max60_1024-coco80-geometry_first-compact_object_box_closed-sorted-packed12k-natural_adjacent-pure_ce/epoch_4-pure_ce-coco80-geometry_first-1024-compact_object_box_closed-sorted-packed12k-natural_adjacent-llm_only/v5-20260617-141636/checkpoint-928
```

Description-first checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/coco_bbox_len12000_1024-coco80-desc_first-compact_object_box_closed-sorted-packed12k-natural_adjacent-pure_ce/epoch_4-pure_ce-coco80-desc_first-1024-coco_bbox_len12000-compact_object_box_closed-sorted-packed12k-natural_adjacent-llm_only/v0-20260617-133335/checkpoint-928
```

The semantic template axis is:

- `detection_template.id` / `custom.detection_template_id`:
  `compact_object_box_closed`
- `custom.object_field_order`: `desc_first` or `geometry_first`
- `custom.object_ordering`: `sorted`

Prompt and parser alignment must be checked before interpreting any probe:
generated prompts, strict parser policy, wrapper token IDs, coordinate-token
IDs, object ordering, image preprocessing, and `do_resize=false` must match the
checkpoint family being probed.

The canonical operational validation source for this study is the COCO
`bbox_len12000` validation JSONL:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

The geometry-first checkpoint name contains `bbox_max60_1024`; do not rewrite
that checkpoint path. For analysis and future reruns, use the `bbox_len12000`
validation source unless a probe explicitly records a narrower comparator.

## Evidence Hierarchy

Existing comparison artifacts are useful for scene discovery and visual
orientation, but not sufficient for mechanism claims unless their prompt,
template, checkpoint, decode, and parser provenance match the probe being
interpreted.

Scene-discovery artifact:

```text
/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/
```

Available `desc_first` `rp=1.10` operational artifact:

```text
/data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase
```

The available `rp=1.10` desc-first artifact supports the qualitative trend,
but has a provenance caveat: its summary records the requested adapter under a
singular `/data/CoordExp/output/...` path even though the currently verified
checkpoint directory is under plural `/data/CoordExp/outputs/...`. Treat this
as an audit item before source-of-truth use.

Existing `geometry_first` diagnostic artifacts used a max60-named validation
root. The first-200 image IDs and GT counts match the available desc-first
`bbox_len12000` artifact, so they may seed cohort selection, but new operational
reruns and hidden-state probes should use the canonical `bbox_len12000` source.

Mechanism claims require fresh or audited operational probes with explicit
metadata for:

- checkpoint path
- template id
- field order
- object ordering
- prompt/template hash or equivalent prompt contract
- decode settings
- parser policy
- image preprocessing
- raw/no-NMS and metric-bearing/guarded scope

## Research Unit

The primary unit is the object-generation step, not final AP or whole-image
success. Each step should be decomposed into:

- prefix state before the object;
- wrapper-token transition;
- description segment or box segment onset;
- coordinate slots `x1`, `y1`, `x2`, `y2`;
- object completion and stop/continuation transition;
- effect of the emitted object on the next autoregressive step.

This unit is deliberately template-aware. In `geometry_first`, spatial binding
may occur before the description. In `desc_first`, semantic willingness may
precede instance binding until coordinate onset.

## Cohort Policy

Use representative and comparable cases, not only visually striking failures.
The first cohort should be selected from audited operational artifacts and may
use existing galleries only as candidate discovery.

The executable pair-study configuration for the first cohort is:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Required buckets:

- clean matched successes;
- desc-first duplication bursts;
- geometry-first duplication bursts;
- significant low-recall / skipped-object cases;
- empty rows or strict-template mismatches;
- same-class crowded scenes;
- dense or tiny-object scenes;
- geometry-first extreme or multi-object boxes.

Keep a held-out reserve so candidate mechanisms can be checked beyond the
examples used to discover them.

## Reusable Prior Evidence And Tooling Policy

The previous autoregressive-duplication branch already established several
mechanistic constraints. This study should reuse those constraints and avoid
spending first-pass effort re-proving them:

- stop-token pressure alone is not an adequate root-cause explanation;
- complete-span log probability alone does not explain duplicate selection;
- coordinate-token locality and embedding geometry can be real while dynamic
  autoregressive coordinate-slot behavior remains fragile;
- false negatives can be recoverable under language-side or coordinate-prefix
  guidance, so visual blindness is not the default explanation;
- attention concentration alone is not enough; useful claims need temporal
  precedence, hidden/logit/attention state, and causal or counterfactual
  movement;
- route selection, value direction, residual compatibility, and sign can split
  across cases, so averaging can hide the mechanism.

Relevant prior synthesis notes:

- `progress/diagnostics/2026-06-12_autoregressive_duplication_causal_chain_synthesis.md`
- `progress/diagnostics/2026-06-12_fn_guidance_and_coord_basin_synthesis.md`
- `progress/diagnostics/2026-06-12_pre_onset_duplication_precursor_synthesis.md`
- `progress/diagnostics/2026-05-18_hard_ce_coord_logit_embedding_locality.md`
- `progress/diagnostics/2026-05-18_gaussian_softce_a5_a6_coord_logit_locality.md`

Tooling from the older
`/data/CoordExp/.worktrees/mechanistic-diagnosis-experiments` branch should be
treated as a reference library. Fetch or port individual helpers only when a
current probe needs them. Do not bulk-port the old random-order checkpoint
configs or phase scripts into this worktree as canonical code.

## Starting Hypothesis Set

The first loop should be creative and progressive, but not detached from prior
evidence. Treat these as competing mechanisms to separate, not conclusions:

1. `desc_first` may increase semantic willingness and object-count pressure by
   entering a class/description branch before instance binding is fully
   spatially grounded.
2. `geometry_first` may bind spatially earlier, but sometimes to a region or
   scene patch rather than an object instance, producing extreme boxes or
   multi-object coverage.
3. Duplicate bursts may start at a local branch where one repeated object has a
   sharper next-token path while remaining valid objects are fragmented across
   descriptions, coordinate basins, or wrapper transitions.
4. Low recall may reflect selection and synchronization failure rather than
   absence of visual representation: the object may be represented, but not
   selected, selected too late, or hidden behind a competing prefix basin.
5. Prefix updates may act as a coverage signal in some contexts and a
   re-attraction signal in others; the sign split is itself a target for
   mechanism discovery.
6. Wrapper-token order may change the layer/token position at which the model
   can commit to an instance, so the same object can have different commitment
   boundaries under `desc_first` and `geometry_first`.

## Mechanism Ambition Standard

The target is a binding operation, not merely a binding correlate. The study
should search for an internal transition that turns visual candidates plus the
current prefix into one selected next object, changes as objects are emitted,
and helps decide whether the next span is a new object, a duplicate, an extreme
region, or stop.

A candidate mechanism should satisfy at least three of these criteria before it
is treated as a serious finding:

- appears before the behavioral divergence;
- predicts selected object or failure type on held-out cases;
- differs systematically between the trained `desc_first` and `geometry_first`
  pair;
- localizes to a token, layer, site, or window with a plausible functional role;
- survives controls against simple probability, attention-mass, object-count,
  bbox-size, and parser-error explanations;
- causal patching or perturbation moves the next-object decision in the
  predicted direction;
- suggests a compact training or inference anchor that is more specific than a
  generic duplicate guard or coverage loss.

Useful mechanism lenses for the first pass:

- commitment boundary: a soft candidate-set phase may precede hard instance
  selection;
- wrapper synchronization gate: structural wrapper tokens may act as control
  gates, not only syntax;
- prefix coverage polarity: emitted objects may suppress their visual source in
  some contexts and attract the next step back in others;
- region-to-instance collapse: `geometry_first` may bind a region before
  resolving, or failing to resolve, a single instance;
- candidate fragmentation: valid remaining objects may fragment over many
  continuations while duplicate branches remain locally sharp;
- object-index clock: sorted ordering may induce an implicit progress state that
  can desynchronize from visual selection;
- coordinate-basin handoff: visual object selection may appear first as a
  coordinate-basin attractor, then transfer into language, or the reverse;
- stop/continue as objectness readout: stopping may reflect whether any
  remaining object has crossed an internal binding threshold.

Early comparative claims must be phrased as differences across a separately
trained checkpoint pair. Do not claim that changing only template order caused a
mechanism until same-checkpoint forced-template or prefix-counterfactual probes
support that stronger causal statement.

## First Operational Probe Family

Begin observationally and escalate to causal tests only after a signal is
localized.

First probes should produce:

- per-step object-span token traces;
- wrapper-token, stop-token, and continuation probabilities;
- coordinate-slot rank/probability/mass around GT and predicted boxes;
- local branch competition among candidate objects;
- teacher-forced versus self-prefix contrasts;
- hidden-state snapshots at pre-commit positions;
- attention and visual-token readouts around candidate object regions,
  duplicate regions, and extreme-box regions;
- prefix perturbation panels that test abrupt basin transitions.

The first deep probe should prioritize commitment-boundary, wrapper-gate, and
remaining-versus-emitted separation signals. For selected object steps, collect
states and logits around:

- before the first object wrapper;
- after wrapper start;
- after description start/end;
- after box start/end;
- before each coordinate slot;
- immediately after object completion;
- before next object or stop.

The strongest early candidate would be a separator or direction that
distinguishes already-emitted GT objects, still-remaining GT objects, duplicate
targets, extreme-region targets, and background/no-object, and that changes
after each generated object.

The available GPU pool for later probe shards is:

```text
0,1,2,3,4,5,6,7
```

Do not spend GPU time until cohort and probe manifests are deterministic.

Only after a candidate signal predicts divergence should the study run causal
patching, ablation, or steering probes. Prefer minimal localized interventions
over broad architectural changes.

## Recording Rule

Record measured results under `progress/diagnostics/` with exact artifact roots,
checkpoint paths, decode settings, parser counters, evidence scope, and
interpretation boundaries. Use `progress/benchmarks/` only when the main result
is a measured checkpoint comparison. Do not promote a training or inference
intervention until the mechanism is supported by observational and causal
evidence.

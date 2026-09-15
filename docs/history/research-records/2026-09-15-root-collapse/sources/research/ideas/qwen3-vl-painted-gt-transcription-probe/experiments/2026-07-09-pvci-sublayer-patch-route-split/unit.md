---
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-sublayer-patch-route-split
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
---

# PVCI Sublayer Patch Route Split

## Scope

This unit follows the residual-patch causal-sufficiency result. Residual
patching showed that the post-scatter visual-token actuator becomes sufficient
at late-middle decoder-layer outputs, but that does not explain which branch
produces the recoverable state.

This probe patches branch outputs inside `Qwen3VLTextDecoderLayer`:

- `self_attn` output before the attention residual add;
- `mlp` output before the MLP residual add.

It intentionally does not claim exact after-attention-residual localization,
because that boundary is not a standalone module in the installed Transformers
implementation.

## Completion Promise

The unit is complete when a branch-output patch probe:

- reuses the same 20 divergence events and strict replay gate;
- patches attention-output and MLP-output branches at the recovered residual
  bands;
- reports step-0 and step-1 separately;
- keeps clean-to-clean no-op branch controls;
- records provenance against the same source feature-store artifacts.

## Evidence Gate

Credible evidence requires:

- no-op branch panels recover approximately zero;
- early control branch panels do not explain the full effect;
- at least one branch type recovers a meaningful fraction of the corresponding
  residual-output patch effect on strict events;
- negative or partial recovery is interpreted as branch-output insufficiency,
  not as evidence that the branch is irrelevant after residual mixing.

## Planned Probe

Script:

```bash
python scripts/probes/painted_gt/run_post_scatter_sublayer_patch_probe.py \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_sublayer_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all \
  --force
```

Panels:

- all events: control window `13-15` for attention and MLP outputs;
- step-0: attention/MLP outputs at layers 17, 19, 21 and window 17-19;
- step-1: attention/MLP outputs at layers 20, 21, 22, 23 and window 21-23;
- clean-to-clean no-op windows for both branch types.

## Executed Evidence

Debug command:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/probes/painted_gt/run_post_scatter_sublayer_patch_probe.py \
  --max-events 4 \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_sublayer_patch/e1_wrong_object_jitter_medium_val32_step0_step1_debug4 \
  --force
```

Full command:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/probes/painted_gt/run_post_scatter_sublayer_patch_probe.py \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_sublayer_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all \
  --force
```

Artifact root:

```text
/data/CoordExp/outputs/painted_gt/pvci_post_scatter_sublayer_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all
```

Main receipts:

- `sublayer_patch_summary.json`
- `sublayer_patch_events.jsonl`
- `sublayer_patch_panels.jsonl`
- `sublayer_patch_report.md`

Scope:

- 20 selected clean-vs-post-scatter divergence events;
- 18 strict margin replay events;
- claim gate: `strict_margin_replay`;
- attention-output and MLP-output hooks score through the real final model head.

Strict-event summary:

| panel | family | strict events | strict mean recovery | strict post-win | strict argmax-post |
| --- | --- | ---: | ---: | ---: | ---: |
| `control_w13_15_attn_final` | all | 18 | 0.1193 | 0.0556 | 0.0556 |
| `control_w13_15_mlp_final` | all | 18 | -0.0069 | 0.0000 | 0.0000 |
| `s0_l17_attn_final` | step0 | 8 | 0.6415 | 0.7500 | 0.7500 |
| `s0_l17_mlp_final` | step0 | 8 | 0.3252 | 0.5000 | 0.5000 |
| `s0_l19_attn_final` | step0 | 8 | 0.1087 | 0.0000 | 0.0000 |
| `s0_l19_mlp_final` | step0 | 8 | 0.4590 | 0.6250 | 0.7500 |
| `s0_w17_19_attn_final` | step0 | 8 | 0.8061 | 0.8750 | 0.8750 |
| `s0_w17_19_mlp_final` | step0 | 8 | 0.7613 | 0.8750 | 0.8750 |
| `s1_l20_attn_final` | step1 | 10 | -0.0188 | 0.0000 | 0.0000 |
| `s1_l21_attn_final` | step1 | 10 | 0.3303 | 0.2000 | 0.2000 |
| `s1_l22_attn_final` | step1 | 10 | -0.1176 | 0.0000 | 0.0000 |
| `s1_l23_attn_final` | step1 | 10 | 0.0444 | 0.0000 | 0.0000 |
| `s1_l22_mlp_final` | step1 | 10 | 0.1594 | 0.0000 | 0.0000 |
| `s1_w21_23_attn_final` | step1 | 10 | 0.3573 | 0.3000 | 0.2000 |
| `s1_w21_23_mlp_final` | step1 | 10 | 0.2356 | 0.0000 | 0.0000 |

## Research Unit Closeout

Observed: Branch-output patching is clearly weaker than full decoder-layer
residual patching. Step-0 stop-vs-continue has partial branch-level recovery:
attention output at layer 17 recovers `0.6415` and flips `0.75` of strict
events, MLP output at layer 19 recovers `0.4590`, and window 17-19 attention or
MLP output reaches roughly `0.76-0.81` recovery with `0.875` strict post-win.
Step-1 description competition is much less branch-output localizable:
single-layer attention/MLP outputs rarely flip the strict two-token winner, and
window 21-23 attention reaches only `0.3573` recovery with `0.3` strict
post-win. Clean-to-clean no-op branch panels recover zero; early MLP control is
near zero and early attention control is low.

Supported: On the 18 strict replay events, raw attention-output or MLP-output
patches alone are not sufficient to reproduce the full recovered decoder-layer
residual effect. Step-0 continuation gating has a partial attention/MLP branch
handle in layers 17-19. Step-1 description identity appears more dependent on
the accumulated residual state than on any single raw branch output tested
here.

Not supported yet: Exact attention-vs-MLP route, after-attention-residual
localization, and whether simultaneous branch-output patching or layer-input
residual patching can explain the recovered residual-output sufficiency.

Next decider: Decompose the recovered full-layer residual patch by state
boundary rather than raw module output alone. The useful next probe is a
layer-boundary residual decomposition: patch layer input, patch after-attention
residual, patch MLP output, patch simultaneous attention-plus-MLP outputs, and
compare with full-layer output patch on the same strict events.

Promotion decision: Not promoted. This remains non-normative research evidence.

---
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-layer-boundary-residual-decomposition
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
---

# PVCI Layer Boundary Residual Decomposition

## Scope

This unit follows two observations:

- decoder-layer output patching is causally sufficient for the immediate next
  divergent token on strict replay events;
- raw attention-output and MLP-output branch patches are weaker, especially for
  step-1 description identity.

The next uncertainty is whether the missing mass is already present in the
layer input residual, appears after the attention residual add, or only becomes
available after the full MLP residual output.

## Completion Promise

The unit is complete when a boundary patch probe:

- reuses the same event-selection and strict replay gate;
- patches `layer_input`, `after_attention_residual`, and `layer_output`;
- reports step-0 and step-1 separately;
- keeps clean-to-clean output no-op controls;
- records event/code/source provenance.

## Evidence Gate

Credible evidence requires:

- output no-op controls recover approximately zero;
- early layer controls remain weak;
- boundary recovery is interpreted relative to the already established full
  layer-output patch result;
- claims remain next-token scoped, not full-row behavior.

## Planned Probe

Script:

```bash
python scripts/probes/painted_gt/run_post_scatter_layer_boundary_patch_probe.py \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_boundary_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all \
  --force
```

Panels:

- control layer 13: input, after-attention residual, output;
- step-0: layers 17, 19, 21 at input, after-attention residual, output;
- step-1: layers 20, 21, 22, 23 at input, after-attention residual, output;
- clean-to-clean output no-op at layer 17 for step-0 and layer 23 for step-1.

## Executed Evidence

Debug command:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/probes/painted_gt/run_post_scatter_layer_boundary_patch_probe.py \
  --max-events 4 \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_boundary_patch/e1_wrong_object_jitter_medium_val32_step0_step1_debug4 \
  --force
```

Full command:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/probes/painted_gt/run_post_scatter_layer_boundary_patch_probe.py \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_boundary_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all \
  --force
```

Artifact root:

```text
/data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_boundary_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all
```

Main receipts:

- `layer_boundary_patch_summary.json`
- `layer_boundary_patch_events.jsonl`
- `layer_boundary_patch_panels.jsonl`
- `layer_boundary_patch_report.md`

Scope:

- 20 selected clean-vs-post-scatter divergence events;
- 18 strict margin replay events;
- claim gate: `strict_margin_replay`;
- wrapped-layer patch points score through the real final model head.

Strict-event summary:

| panel | family | strict events | strict mean recovery | strict post-win | strict argmax-post |
| --- | --- | ---: | ---: | ---: | ---: |
| `control_l13_input_final` | all | 18 | 0.0052 | 0.0000 | 0.0000 |
| `control_l13_after_attn_final` | all | 18 | 0.0088 | 0.0000 | 0.0000 |
| `control_l13_output_final` | all | 18 | 0.0088 | 0.0000 | 0.0000 |
| `s0_l17_input_final` | step0 | 8 | 0.4816 | 0.6250 | 0.6250 |
| `s0_l17_after_attn_final` | step0 | 8 | 0.8932 | 1.0000 | 1.0000 |
| `s0_l17_output_final` | step0 | 8 | 0.8932 | 1.0000 | 1.0000 |
| `s0_l19_input_final` | step0 | 8 | 0.9724 | 1.0000 | 1.0000 |
| `s0_l19_after_attn_final` | step0 | 8 | 0.9782 | 1.0000 | 1.0000 |
| `s0_l19_output_final` | step0 | 8 | 0.9782 | 1.0000 | 1.0000 |
| `s0_l21_input_final` | step0 | 8 | 1.0117 | 1.0000 | 1.0000 |
| `s0_l21_after_attn_final` | step0 | 8 | 1.0059 | 1.0000 | 1.0000 |
| `s0_l21_output_final` | step0 | 8 | 1.0059 | 1.0000 | 1.0000 |
| `s1_l20_input_final` | step1 | 10 | 0.6837 | 0.9000 | 0.9000 |
| `s1_l20_after_attn_final` | step1 | 10 | 0.7534 | 0.9000 | 0.9000 |
| `s1_l20_output_final` | step1 | 10 | 0.7534 | 0.9000 | 0.9000 |
| `s1_l21_input_final` | step1 | 10 | 0.7534 | 0.9000 | 0.9000 |
| `s1_l21_after_attn_final` | step1 | 10 | 0.8299 | 1.0000 | 1.0000 |
| `s1_l21_output_final` | step1 | 10 | 0.8299 | 1.0000 | 1.0000 |
| `s1_l23_input_final` | step1 | 10 | 0.8483 | 1.0000 | 1.0000 |
| `s1_l23_after_attn_final` | step1 | 10 | 0.8865 | 1.0000 | 1.0000 |
| `s1_l23_output_final` | step1 | 10 | 0.8865 | 1.0000 | 1.0000 |

## Research Unit Closeout

Observed: The boundary probe ran successfully and reproduced the full
layer-output residual-patch effect under a wrapped Qwen decoder layer forward.
Layer-13 controls are effectively zero. Step-0 stop-vs-continue is partially
present at layer-17 input (`0.4816` recovery), becomes strongly sufficient
after the layer-17 attention residual (`0.8932`), and is already essentially
fully present at layer-19 input (`0.9724`) and layer-21 input (`1.0117`).
Step-1 description identity is already substantial at layer-20 input
(`0.6837`), reaches stronger sufficiency after layer-20/21 attention residual
updates, and is fully decision-flipping by layer-22/23 input.

Supported: On the 18 strict replay events, the post-scatter steering effect is
carried by the ordinary destination-token residual stream by the recovered
late-middle layers. The raw branch-output probe was weaker because it patched
only branch outputs while leaving the surrounding accumulated residual state
clean. The MLP residual add does not materially change these next-token patch
results at the tested layers: after-attention residual and layer-output panels
are effectively identical.

Not supported yet: Full-row generation behavior under a persistent intervention,
attention-head routing, visual-token-to-destination attention paths, and whether
the same boundary pattern holds for later coordinate-token divergences.

Next decider: The current post-scatter mechanism lane is sufficiently
converged for the augmentation decision: do not prioritize augmentation as a
fix for missing post-scatter actuator entry. If continuing deeper, the next
useful probe is attention-route localization for the layer-17/20 destination
token updates, or a short continuation probe that reapplies the winning
boundary patch across generated steps.

Promotion decision: Not promoted. This remains non-normative research evidence.

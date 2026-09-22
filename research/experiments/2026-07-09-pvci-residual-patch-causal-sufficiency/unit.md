---
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-09-pvci-residual-patch-causal-sufficiency
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
---

# PVCI Residual Patch Causal Sufficiency

## Scope

This unit tests whether the post-scatter visual-token intervention becomes a
causally sufficient decoder residual state for the next divergent token. It
starts from the established post-scatter equivalence and layer-onset probes:

- post-scatter image-token deltas reproduce the earlier `get_image_features`
  image-only delta behavior exactly on val32;
- logit-lens readability appears mainly in late-middle language layers,
  roughly layers 17-21 for step-0 stop-vs-continue and 20-23 for step-1
  description competition.

## Completion Promise

The unit is complete when a paired residual-patch probe:

- replays the same clean and post-scatter divergence events;
- tags strict replay parity before interpreting patch effects;
- patches clean decoder-layer outputs with post-scatter decoder states at
  pre-registered late-middle layers/windows;
- scores patched outputs through the real model head, not a weight-only
  logit lens;
- records provenance for the source feature store, source condition receipts,
  config, and git state.

## Evidence Gate

Credible evidence requires:

- `strict_margin_replay` counts are reported separately from all selected
  events;
- clean-to-clean no-op patch panels remain close to clean behavior;
- negative after-DeepStack control window does not explain the full effect;
- at least one late-middle post-scatter patch panel shows strong margin
  recovery or two-token winner flips on strict events;
- artifacts include event rows, panel rows, summary, and a human-readable
  report.

## Planned Probe

Script:

```bash
python scripts/probes/painted_gt/run_post_scatter_residual_patch_probe.py \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_residual_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all \
  --force
```

Primary panels:

- all events: control window `13-15`, final-position patch;
- step-0 stop-vs-continue: layers 17, 19, 21 and windows 16-18, 17-19,
  19-21;
- step-1 description competition: layers 20, 21, 22, 23 and windows 20-22,
  21-23;
- positive-control windows use `full_sequence`;
- clean-to-clean no-op windows use final-position patches.

## Executed Evidence

Debug command:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/probes/painted_gt/run_post_scatter_residual_patch_probe.py \
  --max-events 4 \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_residual_patch/e1_wrong_object_jitter_medium_val32_step0_step1_debug4 \
  --force
```

Full command:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/probes/painted_gt/run_post_scatter_residual_patch_probe.py \
  --output-root /data/CoordExp/outputs/painted_gt/pvci_post_scatter_residual_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all \
  --force
```

Artifact root:

```text
/data/CoordExp/outputs/painted_gt/pvci_post_scatter_residual_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all
```

Main receipts:

- `residual_patch_summary.json`
- `residual_patch_events.jsonl`
- `residual_patch_panels.jsonl`
- `residual_patch_report.md`

Scope:

- 20 clean-vs-post-scatter divergence events from the val32 PVCI wrong-object
  jitter-medium source root;
- 18 strict margin replay events;
- the causal-sufficiency claim denominator is the 18 strict replay events, not
  all 20 selected events;
- step-0 stop-vs-continue: 9 selected, 8 strict;
- step-1 description competition: 11 selected, 10 strict.

Strict-event summary:

| panel | family | strict events | strict mean recovery | strict post-win | strict argmax-post |
| --- | --- | ---: | ---: | ---: | ---: |
| `control_w13_15_final` | all | 18 | 0.1300 | 0.0556 | 0.0556 |
| `s0_l17_final` | step0 | 8 | 0.8932 | 1.0000 | 1.0000 |
| `s0_l19_final` | step0 | 8 | 0.9782 | 1.0000 | 1.0000 |
| `s0_l21_final` | step0 | 8 | 1.0059 | 1.0000 | 1.0000 |
| `s0_w17_19_clean_noop` | step0 | 8 | 0.0000 | 0.0000 | 0.0000 |
| `s0_w17_19_final` | step0 | 8 | 0.9782 | 1.0000 | 1.0000 |
| `s0_w17_19_full` | step0 | 8 | 1.0000 | 1.0000 | 1.0000 |
| `s1_l20_final` | step1 | 10 | 0.7534 | 0.9000 | 0.9000 |
| `s1_l21_final` | step1 | 10 | 0.8299 | 1.0000 | 1.0000 |
| `s1_l22_final` | step1 | 10 | 0.8483 | 1.0000 | 1.0000 |
| `s1_l23_final` | step1 | 10 | 0.8865 | 1.0000 | 1.0000 |
| `s1_w21_23_clean_noop` | step1 | 10 | 0.0000 | 0.0000 | 0.0000 |
| `s1_w21_23_final` | step1 | 10 | 0.8865 | 1.0000 | 1.0000 |
| `s1_w21_23_full` | step1 | 10 | 1.0000 | 1.0000 | 1.0000 |

## Interpretation Rules

- A positive patch result supports causal sufficiency for the patched decoder
  residual boundary, not attention/MLP localization.
- A negative final-position result with positive full-sequence recovery means
  the mediator is not only the destination token residual at that layer.
- Final-position multi-layer windows are last-patched-boundary checks, not
  independent cumulative window evidence, because later hooks overwrite the
  same destination-token state.
- Attention/MLP decomposition should wait until one residual boundary recovers
  enough of the post-scatter next-token decision.

## Research Unit Closeout

Observed: The residual-patch probe ran on 20 selected step-0/step-1 divergence
events and found 18 strict margin replay events. Clean-to-clean no-op panels
had zero recovery. The after-DeepStack negative control window `13-15` had low
strict recovery (`0.1300`) and almost never flipped the two-token decision.
Late-middle post-scatter decoder residual patches were strongly causal:
step-0 final-position patching at layer 17 already recovered `0.8932`, and
layers 19/21 recovered approximately full post-scatter behavior. Step-1
description competition recovered `0.7534` at layer 20 and strengthened through
layer 23 (`0.8865`), while full-sequence layer 21-23 patching reached `1.0`.

Supported: On the 18 strict replay events from this val32 divergence-selected
PVCI slice, the post-scatter visual-token actuator becomes a causally
sufficient late-middle decoder residual state for the immediate next divergent
token. The two non-strict selected events are excluded from the
causal-sufficiency denominator. The supported split is: stop-vs-continue
control is already destination-token-residual sufficient by roughly layer 17,
while description identity selection is later and ramps through roughly layers
20-23.

Not supported yet: Attention/MLP localization, the exact sublayer source of the
state transition, visual-token-stream mediation versus destination-token
mediation before the recovered layers, and whether the same pattern holds for
later coordinate divergences beyond step 0/1.

Next decider: Split the recovered residual bands by sublayer. For step-0, start
with layers 17/19/21 or windows 17-19 and 19-21. For step-1, start with layers
20-23 or window 21-23. The next probe should distinguish attention-output
patching, MLP-output patching, and residual addition points before making a
route-level claim.

Promotion decision: Not promoted. This remains non-normative research evidence.

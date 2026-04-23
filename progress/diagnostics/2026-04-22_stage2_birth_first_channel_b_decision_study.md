---
title: Stage-2 Birth-First Channel-B Decision Study
date: 2026-04-22
status: consolidated-final
owner: codex
depends_on:
  - openspec/changes/birth-first-stage2-channel-b/design.md
  - docs/superpowers/specs/2026-04-22-birth-first-stage2-channel-b-design.md
  - progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/control_vllm_bonly_merged1332_logging.jsonl
  - progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/enabled_smallfrac_logging.jsonl
  - progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/control_smallfrac_logging.jsonl
  - progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/enabled_smallfrac_launch.log
  - progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/control_smallfrac_launch.log
---

# Stage-2 Birth-First Channel-B Decision Study

## Why This Note Exists

This note records the current research-facing status of the birth-first
Stage-2 Channel-B direction after the first complete operator/debugging loop and
the first paired small-fraction vLLM decision study.

It is meant to answer three questions cleanly:

1. what was required to make the vLLM study surface trustworthy;
2. what the paired control-versus-birth-first run actually showed;
3. what the next justified move is.

This is a diagnostics note, not a benchmark note, because the main result is a
mechanism and decision read rather than a new SOTA claim.

## Fixed Study Contract

The whole study was intentionally pinned to one checkpoint family:

- base model:
  `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
- Stage-1 adapter checkpoint:
  `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`

The vLLM server-side study surface was converted to a merged full model through:

- merge script:
  `/data/CoordExp/scripts/merge_coord.sh`
- merged checkpoint:
  `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`

The intended algorithmic change under test stayed narrow:

- same anchor-first, clean-prefix, one-forward Stage-2 shape
- same `K=2` anchor-plus-one-explorer profile
- same merged full checkpoint on both arms
- only the `birth_first` Channel-B behavior differs between control and enabled

## Artifact Bundle

### Spec / design surface

- OpenSpec design:
  `/data/CoordExp/openspec/changes/birth-first-stage2-channel-b/design.md`
- superpower design note:
  `/data/CoordExp/docs/superpowers/specs/2026-04-22-birth-first-stage2-channel-b-design.md`

### Stable evidence bundle

The original runs lived in a transient worktree. The durable evidence copied
into the main repo for this note is:

- bundle root:
  `/data/CoordExp/progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/`

### Operator validation surface

- merged-model B-only vLLM smoke:
  `/data/CoordExp/progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/control_vllm_bonly_merged1332_logging.jsonl`

### Paired decision study

- enabled run:
  `/data/CoordExp/progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/enabled_smallfrac_logging.jsonl`
- control run:
  `/data/CoordExp/progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/control_smallfrac_logging.jsonl`

### Paired launch logs

- enabled:
  `/data/CoordExp/progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/enabled_smallfrac_launch.log`
- control:
  `/data/CoordExp/progress/diagnostics/artifacts/stage2_birth_first_channel_b_decision_study_2026-04-22/control_smallfrac_launch.log`

## Main Reliable Results

### 1. The original vLLM failure was primarily a model-surface mismatch

Before the merge, the learner was using the Stage-1 adapter surface while the
vLLM rollout server was serving only the plain base model. That mismatch
produced catastrophic rollout pathology:

- `N_valid_pred = 0`
- very large `wrong_arity`
- very large truncation
- unusable Channel-B rollouts

After switching the vLLM server to the merged full checkpoint, the same
`3 server + 1 learner` topology became scientifically usable:

- valid predictions reappeared
- truncation dropped substantially
- rollout F1 became nonzero
- the full vLLM operator path completed cleanly

This is now the strongest operator conclusion from the study:

> for this Stage-2 vLLM decision surface, using a merged full model is
> effectively required.

### 2. The current birth-first variant is real, but not a clean win

The paired small-fraction study used:

- `8` train steps
- `768` train samples
- `128` eval samples
- merged full checkpoint on both arms
- same vLLM topology and prompt/geometry surface

#### Train-side rollout view at final step

| metric | enabled | control |
| --- | ---: | ---: |
| rollout precision | `0.6827` | `0.7895` |
| rollout recall | `0.2970` | `0.2889` |
| rollout F1 | `0.4139` | `0.4230` |
| fp / sample | `0.8958` | `0.5000` |
| fn / sample | `4.5625` | `4.6146` |
| valid pred count | `271` | `228` |
| invalid drop count | `284` | `185` |
| wrong-arity drops | `162` | `142` |
| bbox-invalid drops | `107` | `37` |
| duplicate objects suppressed | `48` | `20` |
| near-IoU90 same-desc pairs | `16` | `38` |
| gating rejection rate | `0.4841` | `0.5360` |
| recovered GT count | `28` | `25` |

#### Eval view on the 128-sample slice

| metric | enabled | control |
| --- | ---: | ---: |
| eval precision | `0.6245` | `0.6800` |
| eval recall | `0.6759` | `0.6648` |
| eval F1 | `0.6492` | `0.6723` |
| eval mAP | `0.4514` | `0.4494` |
| eval FP total | `365` | `281` |
| eval FN total | `291` | `301` |
| eval parse dropped invalid | `116` | `2` |
| eval parse truncated rate | `0.0078` | `0.0` |

This means the enabled birth-first arm did **not** collapse or noop. It
activated the intended mechanism and bought some recall-side movement, but it
did not yet produce a clean overall improvement.

### 3. The best-supported interpretation is “too permissive”, not “idea failed”

The current enabled arm shows several signals in the desired direction:

- slightly higher recall
- lower gating rejection rate
- lower duplicate-like near-neighbor counts
- nonzero continue-over-EOS boundaries
- dead anchors driven to zero

But it also shows clear costs:

- lower precision
- worse eval F1
- much higher malformed / invalid prediction burden

The most informative bucket shift is:

- enabled:
  - `dead_anchor_count = 0`
  - `neutral_shielded_count = 33`
- control:
  - `dead_anchor_count = 26`
  - `neutral_shielded_count = 0`

That suggests the current birth-first configuration is not just improving stop
calibration. It is also relaxing triage enough that too many weak or malformed
hypotheses remain live.

So the current read is:

> birth-first is directionally alive, but the present variant is too permissive.

### 4. Control is the safer promotion candidate; birth-first remains the better research candidate

If the immediate question is “which current recipe should be trusted more for a
larger stable run,” the answer is:

- **control**

If the question is “which direction still looks worth iterating on for the main
research topic,” the answer is:

- **birth-first**

but with a stricter next version, not this exact enabled configuration.

## Current Decision

The current enabled birth-first configuration should **not** be promoted
directly to a larger real-training run.

The evidence supports:

1. keeping the merged-vLLM operator path;
2. keeping the birth-first research direction;
3. running one more small-fraction birth-first ablation before any full-budget
   promotion.

## Recommended Next Step

The next study should be a narrower birth-first v2, not a broader machinery
jump.

The cleanest justified follow-up is:

- keep the same merged-vLLM setup;
- keep the same small-fraction paired protocol;
- preserve the current control arm;
- tighten the enabled arm so that birth-first / EOS pressure is retained but
  dead-anchor handling is stricter.

The most likely high-value adjustment is:

- reduce the current relaxation that turns many would-be dead anchors into
  `neutral_shielded` live hypotheses.

A secondary possible adjustment is:

- weaken `continue_over_eos` slightly after the triage tightening, if malformed
  continuation remains too easy.

## Bottom-Line Summary

The reliable conclusions from this whole loop are:

1. merged full model for vLLM is required on this study surface;
2. the current birth-first arm is not a no-op and does improve some recall-side
   behavior;
3. the current birth-first arm is still too loose because it increases malformed
   and weak continuation leakage;
4. control is the safer current recipe, but birth-first remains the right
   research direction for the next targeted iteration.

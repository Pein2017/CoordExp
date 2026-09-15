---
title: Training and Runtime Lessons
type: investigation
role: lessons
authority: non_normative_research
status: historical-synthesis
updated: 2026-07-17
---

# Training and Runtime Lessons

These are operational lessons extracted from historical runs. They answer
“what should a probe verify first?” rather than defining current behavior.
Current training ownership remains in [`docs/training/README.md`](../../../docs/training/README.md),
[`docs/training/STAGE2_RUNBOOK.md`](../../../docs/training/STAGE2_RUNBOOK.md),
and the stable OpenSpec contracts.

## Lessons

| ID | Symptom / tempting inference | Evidence handle | Lesson and cheap discriminator | Current owner / stale risk |
|---|---|---|---|---|
| TRAIN-001 | Similar teacher-forcing loss curves are treated as evidence of similar detector quality. | `progress/benchmarks/stage1_training_dynamics_4b_2026-02-26.md`; three 4B Stage-1 runs and 200-row rollout results. | Always pair training scalars with a small free decode. Record object count, stop/truncation, parser validity, duplicates, and AP/F1. Teacher forcing and free rollout answer different questions. | `docs/training/METRICS.md` and `docs/eval/WORKFLOW.md`; historical metrics may use older schemas. |
| TRAIN-002 | A training objective change is credited for a metric delta while decode policy changed too. | `progress/benchmarks/2026-05-07_stage1_2b_coord_component_rp110_ablation.md`; hard CE RP1.05 vs RP1.10. | Treat decode kwargs as an experimental factor. Pin temperature, top-p, repetition penalty, max tokens, stop/EOS, batch topology, and seed before attributing an objective effect. | Current backend/config determines effective kwargs; old config-only knobs are stale unless receipt-backed. |
| TRAIN-003 | A nominal 768/1024 data preset is interpreted as a resolution ablation. | `progress/stage1_coco80_eval_4b_ckpts_768_vs_1024_2026-02-26.md`. | Check actual image dimensions and rescale manifests before interpreting a preset. The historical COCO rows used the same 640×640 images, so deltas were mostly stochasticity/batch effects. | Current public-data provenance and data docs; historical preset names are not enough. |
| TRAIN-004 | A compact/full or packed run is assumed equivalent to a JSON coordinate-token run. | `progress/benchmarks/2026-07-03_coord_token_val200_benchmark.md`, especially its cross-surface caveat. | Record template, wrapper, serialization, ordering, packing, coordinate surface, and max length as first-class identity. Compare only within a declared group. | Current training router and packing docs; revalidate surface support in the target worktree. |
| TRAIN-005 | A checkpoint README/model card is treated as a complete training receipt. | `outputs/stage1_2b/**/checkpoint-*/README.md` and `outputs/probes/coordexp_swift/dora_roundtrip/README.md`. | Generic model cards often contain only base model, PEFT metadata, and placeholders. Require resolved config, runtime receipt, data hash, checkpoint identity, and artifact root before making a training claim. | Current training artifact contract; model-card READMEs remain receipt-only. |
| TRAIN-006 | A parser or helper correction is applied after the fact and old derived metrics remain pooled. | `progress/diagnostics/2026-03-26_stage2_small_object_duplication_offline_synthesis.md`; `v2_bboxfix`. | A diagnostics fix creates a new trust boundary. Freeze the corrected subset, record the code/version/hash, supersede earlier derived rows, and rerun any cross-run comparison. | Current artifact/analysis owner; historical corrected roots require explicit mapping. |
| TRAIN-007 | A smoke passes and is promoted to a production or scientific conclusion. | Painted-GT OpenSpec change and historical smoke/model-card outputs; current research units mark smoke evidence separately. | Smoke proves wiring only: config load, model setup, one forward/backward, artifact write, and parser/row binding. Promotion requires the declared gate, denominators, controls, and population scope. | Current smoke specs and research-unit contracts; no smoke-only architecture claim. |
| TRAIN-008 | Old Stage-2 `stage2_ab` knobs are copied into current experiments. | `progress/benchmarks/2026-02-01_stage2_channel_a_infer_eval.md`; current `docs/training/STAGE2_RUNBOOK.md`. | Translate historical findings onto the current `stage2_two_channel` owner. If no translation has been executed, mark the lesson historical and do not prescribe the old knob. | `docs/training/STAGE2_RUNBOOK.md`; stale-risk is high for removed/deprecated fields. |
| TRAIN-009 | Adapter/export identity is omitted because a checkpoint directory name looks self-explanatory. | `progress/benchmarks/2026-04-21_mixed_objective_sota_checkpoint_probe.md` and Stage-1 benchmark rows. | Record base model, adapter vs merged export, token embedding delta, target modules, config hash, and exact checkpoint. A path alone is not evidence. | Current training artifact contracts; legacy remote/output roots may be pruned. |

## Required lesson receipt

Every new training lesson should include: symptom, mistaken inference, bounded
source, artifact scope, cheap discriminator, safe action, current owner link,
and stale-risk label. Do not add current CLI commands or duplicate schema text
here; link the owner instead.

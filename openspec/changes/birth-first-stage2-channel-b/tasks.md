## 1. Contract And Config Surface

- [ ] 1.1 Update the Stage-2 OpenSpec deltas and typed schema so `stage2_ab.channel_b.birth_first` is validated with `enabled`, `continue_over_eos_weight`, and `continue_over_eos_margin`.
- [ ] 1.2 Enforce the birth-first cross-field rules: `pseudo_positive.enabled=false`, `triage_posterior.num_rollouts=2`, and non-zero `token_ce.config.rollout_global_prefix_struct_ce_weight`.
- [ ] 1.3 Update `docs/training/STAGE2_RUNBOOK.md` to pin the study to the fixed base model `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp` plus the fixed adapter checkpoint `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`, document the `birth_first` knob set, state that birth-first reuses the existing `K=2` control profile, and spell out the paired decision-run protocol.
- [ ] 1.4 Update `docs/training/METRICS.md` to document the birth-first metric family, including additive counters, recovered-ground-truth numerator/denominator surfaces, and `continue_over_eos_margin` as a mean-like rollout-text atom.

## 2. Channel-B Runtime And Objective Routing

- [ ] 2.1 Partition retained unmatched anchor objects into `support_positive_shielded` and `neutral_shielded` under the existing anchor-plus-one-explorer association path.
- [ ] 2.2 Route `support_positive_shielded` objects into rollout-prefix structure CE in birth-first mode while keeping them outside extra desc CE and positive geometry/coord supervision.
- [ ] 2.3 Keep recovered GT objects on the weighted FN-injection path and add one local continue-over-EOS margin target per eligible recovered boundary.
- [ ] 2.4 Preserve the existing duplicate-burst unlikelihood path unchanged as the narrow B-only guardrail in birth-first mode.

## 3. Metrics, Tests, And Reproducibility

- [ ] 3.1 Emit `train/triage/support_positive_shielded_count`, `train/triage/neutral_shielded_count`, `train/triage/recovered_ground_truth_count`, `train/triage/recovered_ground_truth_rate_num`, `train/triage/recovered_ground_truth_rate_den`, `stage2_ab/channel_b/birth_first/N_continue_over_eos_boundaries`, `stage2_ab/channel_b/birth_first/N_continue_over_eos_skipped_no_recovered_boundary`, and `loss/B_rollout_text/continue_over_eos_margin`.
- [ ] 3.2 Add or update focused tests for schema validation, retained-anchor partitioning, one-forward continue-over-EOS projection, and additive-vs-mean metric aggregation.
- [ ] 3.3 Run targeted validation commands and record results in the change notes:
  - `conda run -n ms python -m pytest tests/test_stage2_ab_channel_b_pipeline_guard.py tests/test_stage2_ab_config_contract.py -q`
  - plus any new focused birth-first tests added by this change.

## 4. Decision Runs And Promotion

- [ ] 4.1 Author paired `K=2` configs rooted in the fixed base model `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp` plus the fixed adapter checkpoint `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`:
  - `configs/stage2_two_channel/smoke/birth_first_k2_control_4steps.yaml`
  - `configs/stage2_two_channel/smoke/birth_first_k2_enabled_4steps.yaml`
  - `configs/stage2_two_channel/prod/birth_first_k2_control_decision.yaml`
  - `configs/stage2_two_channel/prod/birth_first_k2_enabled_decision.yaml`
- [ ] 4.2 Run the short decision round against the paired `birth_first_k2_control_decision.yaml` and `birth_first_k2_enabled_decision.yaml` configs; they MUST differ only in documented birth-first knobs plus run-identifying fields such as output path or run name.
- [ ] 4.3 Keep the paired decision runs matched on seed, max steps, eval cadence, data surface, rollout mode, fixed `model.model` base path, and the single fixed adapter checkpoint path in `model.adapters`, and cite `resolved_config.json` plus `experiment_manifest.json` for both runs in the comparison note.
- [ ] 4.4 Compare recall, recovered-GT learning using both count and rate numerator/denominator surfaces, continue-over-EOS activation, duplicate burden via the existing `dup/raw/*` and `stage2_ab/channel_b/dup/N_*` metrics, and rollout stability before promoting the winning `K=2` variant to a long real-training run.

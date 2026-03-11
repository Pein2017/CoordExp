## 1. OpenSpec and Contract Foundation

- [ ] 1.1 Add delta spec files for:
  - `stage2-ab-training`
  - `rollout-matching-sft`
  - `teacher-forcing-objective-pipeline`
  - `teacher-forcing-unified-loss-registry`
  - `trainer-metrics-components`
- [ ] 1.2 Update `progress/directions/full_idea_v3.md` so the locked v1 contract is explicit:
  - anchor-edited target
  - single merged forward
  - explorer-only non-GT-backed -> dead by default
  - recovered GT weighting only, no recovered-prefix distillation in v1
- [ ] 1.3 Validate the new change artifacts with `openspec validate --type change stage2-b-v3-k2-triage --strict --no-interactive`.

## 2. Typed Config and Rollout Contract

- [ ] 2.1 Extend `src/config/schema.py` so `stage2_ab.channel_b` accepts the grouped v3 block:
  - `v3_k2.explorer_temperature`
  - `v3_k2.explorer_top_p`
  - `v3_k2.explorer_top_k`
  - `v3_k2.consistent_iou_threshold`
  - `v3_k2.recovered_fn_weight`
- [ ] 2.2 Keep existing `stage2_ab.channel_b` knobs intact:
  - `duplicate_iou_threshold`
  - `producer_wait_timeout_s`
  - `ddp_phase_timeout_s`
- [ ] 2.3 Add a per-call rollout decode override seam that works across both HF and vLLM backends:
  - anchor greedy override
  - explorer stochastic override
  - fail-fast when a backend/runtime cannot honor the requested policy
- [ ] 2.4 Update canonical Stage-2 YAML profiles under `configs/stage2_two_channel/` to author the new v3 fields explicitly for the selected implementation profile.
- [ ] 2.5 Add config-contract and rollout-contract tests covering accepted keys, missing required fields, rejection of misplaced legacy keys, and dual-policy decode support.

## 3. Channel-B Dual-Rollout Preparation

- [ ] 3.1 Extend `Stage2ABTrainingTrainer._prepare_batch_inputs_b` in `src/trainers/stage2_two_channel.py` to request two rollout views per sample:
  - anchor greedy
  - explorer stochastic
- [ ] 3.2 Reuse the existing per-run v2 cleanup path independently for each rollout:
  - bounded salvage + strict acceptance
  - bbox-valid filtering
  - sequential dedup
  - Hungarian matching
- [ ] 3.3 Add a deterministic one-to-one max-IoU anchor/explorer association helper with stable tie-breaks.
- [ ] 3.3a Add a unit case where a `2x2` association problem has equal IoU totals across multiple assignments and verify the lexicographically smallest sorted pair list is chosen.
- [ ] 3.4 Project association results into side-specific action states:
  - `anchor_gt_backed`
  - `recovered_fn`
  - `shielded_anchor`
  - `dead_anchor`
  - `dead_explorer`
- [ ] 3.5 Keep DDP/runtime control flow symmetric across ranks for the dual-rollout path.

## 4. Target Construction and Weighting

- [ ] 4.1 Build the final teacher-forced target as an **anchor-edited** clean sequence, preserving anchor order.
- [ ] 4.2 Keep shielded anchor objects as neutral context only.
- [ ] 4.3 Remove dead anchor objects from the positive prefix.
- [ ] 4.4 Detect recovered GT objects as:
  - missed in anchor
  - hit in explorer
- [ ] 4.5 Keep recovered GTs on the existing FN-injection path and add the configured higher **per-object desc+geo+coord** weight.
- [ ] 4.6 Thread recovered-FN metadata through the rollout segment payload so `token_ce`, `bbox_geo`, and `coord_reg` can distinguish recovered FN objects from ordinary FN objects.
- [ ] 4.7 Do not add recovered-prefix distillation or any second teacher-forced payload in this implementation slice.

## 5. Local Negative Supervision

- [ ] 5.1 Broaden the upstream source of `duplicate_ul_targets` so it can represent any dead anchor-side continuation, not only same-desc duplicate bursts.
- [ ] 5.2 Keep the first-divergence / LCP suppression mechanism and the `duplicate_ul` module name.
- [ ] 5.3 Restrict dead-continuation UL to anchor-side continuations only.
- [ ] 5.4 Add unit coverage for:
  - anchor-only vs explorer-only GT hits projecting to the correct side-specific action state
  - shielded objects producing no positive CE/geo
  - explorer-only dead objects producing no separate explore-side branch
  - dead anchor continuations creating the expected UL targets
  - recovered FN weighting being applied only to recovered GT tail objects across desc+geo+coord

## 6. Metrics, Dumps, and Docs

- [ ] 6.1 Add triage count metrics:
  - `stage2_ab/channel_b/triage/N_anchor_gt_backed`
  - `stage2_ab/channel_b/triage/N_shielded_anchor`
  - `stage2_ab/channel_b/triage/N_dead_anchor`
  - `stage2_ab/channel_b/triage/N_dead_explorer`
  - `stage2_ab/channel_b/triage/N_recovered_gt`
- [ ] 6.2 Add aggregation-safe triage numerators / denominators:
  - `stage2_ab/channel_b/triage/recovered_gt_num`
  - `stage2_ab/channel_b/triage/recovered_gt_den`
  - `stage2_ab/channel_b/triage/dead_anchor_num`
  - `stage2_ab/channel_b/triage/dead_anchor_den`
- [ ] 6.3 Preserve legacy duplicate metrics as supporting diagnostics and update monitor dumps to surface both rollout views plus final triage decisions.
- [ ] 6.4 Update docs after implementation:
  - `docs/training/STAGE2_DESIGN.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
  - `docs/IMPLEMENTATION_MAP.md` if routing guidance changes

## 7. Validation

- [ ] 7.1 Run targeted config and trainer unit tests:
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_objective_atoms_projection.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_pending_metrics_aggregation.py`
- [ ] 7.2 Run a config materialization check for the selected Stage-2 profile with the new v3 fields.
- [ ] 7.3 Run a short Stage-2 smoke that exercises the dual-rollout B-step path and verifies:
  - no DDP/control-flow divergence,
  - non-zero triage metrics,
  - recovered-FN weighting appears in desc+geo+coord objective paths,
  - dead-anchor UL targets are emitted.
- [ ] 7.4 Validate the OpenSpec change again before implementation handoff:
  - `openspec validate --type change stage2-b-v3-k2-triage --strict --no-interactive`

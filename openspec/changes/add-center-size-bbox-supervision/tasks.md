## 1. Config Surface

- [x] 1.1 Extend typed bbox geometry config validation so `custom.bbox_geo`, Stage-2 `bbox_geo.config`, and the Stage-2 objective-module allowlist in `src/trainers/teacher_forcing/module_registry.py` accept `parameterization`, `center_weight`, and `size_weight` with fail-fast guards.
- [x] 1.2 Add strict validation for `parameterization: xyxy | center_size` and reject `center_size` configs that set both `center_weight` and `size_weight` to zero.
- [x] 1.3 Update the relevant config-contract tests, including `tests/test_stage2_ab_config_contract.py`, `tests/test_training_config_strict_unknown_keys.py`, and rollout-matching schema coverage, so the new bbox-geometry keys are accepted canonically and unknown aliases still fail fast.

## 2. BBox Geo Implementation

- [x] 2.1 When touching both bbox geometry paths, factor the common regression decomposition and reduction into a minimal shared helper so Stage-1 and Stage-2 do not carry separate `center_size` math.
- [x] 2.2 Refactor the Stage-1 bbox geometry path in `src/trainers/losses/bbox_geo.py` and its host wiring in `src/trainers/metrics/mixins.py` so decoded predicted and GT boxes stay canonical `xyxy` outwardly but contribute center-wise `(cx, cy)` and soft size `(log_w, log_h)` terms when `parameterization: center_size` is enabled.
- [x] 2.3 Refactor the Stage-2 bbox geometry path in `src/trainers/teacher_forcing/modules/bbox_geo.py` so decoded predicted and GT boxes stay canonical `xyxy` outwardly but contribute center-wise `(cx, cy)` and soft size `(log_w, log_h)` terms when `parameterization: center_size` is enabled.
- [x] 2.4 Keep CIoU on canonical `xyxy` boxes, preserve existing loss key names, and preserve the published canonical bbox state payload consumed by `bbox_size_aux`, `coord_reg`, and `coord_diag`.

## 3. Verification

- [x] 3.1 Add unit tests for the shared bbox regression helper plus `src/trainers/teacher_forcing/modules/bbox_geo.py` and `src/trainers/losses/bbox_geo.py`, covering default `xyxy` behavior, center-wise loss-space behavior, and epsilon-guarded `log_w` / `log_h` handling.
- [x] 3.2 Add or update tests that prove existing bbox geometry metric keys remain stable while `bbox_smoothl1` semantics follow the resolved parameterization, and that legacy two-key bbox_geo configs still resolve to default `xyxy`.
- [x] 3.3 Add at least one focused parity-style test for the shared regression helper proving it applies the same decomposition and reduction semantics for equivalent canonical decoded boxes and weights, then keep thin wrapper-level coverage showing both Stage-1 and Stage-2 route into those semantics.
- [x] 3.4 Run targeted Stage-1 validation with `conda run -n ms python -m pytest tests/test_bbox_size_aux_loss.py tests/test_training_config_strict_unknown_keys.py tests/test_stage1_static_packing_runtime_config.py`.
- [x] 3.5 Run targeted Stage-2 validation with `conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py tests/test_stage2_objective_atoms_projection.py tests/test_teacher_forcing_loss_catalog.py` plus the rollout-matching schema/config test file that covers `bbox_geo.config`.

## 4. Docs And Reproducibility

- [x] 4.1 Update the Stage-1 objective doc and Stage-2 runbook to document that center-size supervision is an internal experimental loss mode that preserves canonical external `bbox_2d` / `xyxy` contracts.
- [x] 4.2 Update metric-interpretation docs so `bbox_smoothl1` is documented as the configured bbox regression term while `bbox_ciou` remains canonical `xyxy`.
- [x] 4.3 Add an eval-facing note that the base/scored artifact split and `resolved_config.path`-based recovery rules remain unchanged by this training-only parameterization.
- [x] 4.4 Add one smoke-ready Stage-2 experimental config or documented command with an explicit `training.run_name`, and verify the resulting run artifacts include `resolved_config.json` reflecting `parameterization: center_size`.
  Verified a real single-GPU direct learner smoke in the worktree with `gpus=0 config=configs/stage2_two_channel/smoke/a_only_center_size_2steps.yaml conda run -n ms bash scripts/train.sh`. The run completed `2/2` steps under `output/stage2_ab/smoke/a_only_center_size_2steps/smoke_2steps-stage2-a_only-center_size_bbox_geo/v0-20260403-072300/`, produced `resolved_config.json`, `pipeline_manifest.json`, `runtime_env.json`, `effective_runtime.json`, `run_metadata.json`, and `logging.jsonl`, and the resolved bbox config preserved `parameterization = center_size`, `center_weight = 1.0`, and `size_weight = 0.25`. The emitted training log also included non-zero `loss/coord/bbox_smoothl1 = 0.08654976`, `loss/coord/bbox_ciou = 0.09458837`, and `loss/coord/bbox_log_wh = 0.01682876` at step `1/2`.
- [x] 4.5 Optionally add one Stage-1 smoke config or documented snippet as a cheaper debug loop, but do not treat it as the only smoke evidence for this change.

## 1. Lock V1 Scope In Artifacts

- [ ] 1.1 Encode the locked Stage-1 bbox-only assumption in the spec and docs.
- [ ] 1.2 Encode immediate optional Channel-A `A1` support for `bbox_size_aux`.
- [ ] 1.3 Validate the draft change artifacts:
  - `openspec validate --type change bbox-log-size-aux-loss --strict --no-interactive`

## 2. Shared Geometry Helper

- [ ] 2.1 Add shared decoded-box size-aux helper(s) in
      `src/trainers/teacher_forcing/geometry.py`:
  - matched log-width/log-height loss
  - matched log-area loss
  - optional oversize penalty
- [ ] 2.2 Keep helper behavior numerically stable:
  - canonicalize first
  - clamp width/height with `eps`
  - guard empty masks
  - `nan_to_num` returned scalars
- [ ] 2.3 Add direct unit coverage for:
  - exact-match near-zero loss
  - positive width/height mismatch
  - canonicalization with reversed corners
  - empty/invalid masks
  - oversize thresholded zero-vs-positive behavior

## 3. Stage-2 Pipeline Wiring

- [ ] 3.1 Add a new pipeline objective module `bbox_size_aux` to the strict
      registry / allowlists / manifest validation.
- [ ] 3.2 Reuse the shared helper through a new
      `src/trainers/teacher_forcing/modules/bbox_size_aux.py` plugin so matched
      decoded boxes contribute:
  - `bbox_log_wh`
  - `bbox_log_area`
  - `bbox_oversize` when enabled
- [ ] 3.3 Thread decoded canonicalized bbox state from `bbox_geo` into
      `bbox_size_aux` so the new plugin does not reimplement a second decode
      path.
- [ ] 3.4 Update Stage-2 objective-atom projection / logging so the new
      size-aux terms are emitted explicitly.
- [ ] 3.5 Update canonical Stage-2 YAML examples under
      `configs/stage2_two_channel/` with conservative defaults:
  - add `bbox_size_aux` after `bbox_geo`
  - `log_wh_weight > 0`
  - `log_area_weight = 0`
  - `oversize_penalty_weight = 0`

## 4. Rollout-Aligned Reuse

- [ ] 4.1 Ensure `rollout_matching.pipeline.objective[*].name=bbox_size_aux`
      is supported with the same config keys.
- [ ] 4.2 Verify rollout-aligned matched teacher-forcing supervision reuses the
      same `bbox_size_aux` plugin implementation without forked loss logic.

## 5. Stage-1 Wiring

- [ ] 5.1 Add a typed Stage-1 config block under `custom.bbox_size_aux`.
- [ ] 5.2 Add a lightweight Stage-1 aux plugin host around the existing single
      forward / reporter flow, reusing the canonical `bbox_size_aux` plugin
      implementation instead of another feature-specific mixin.
- [ ] 5.3 Reuse the shared decoded-box helper once bbox grouping is explicit.
- [ ] 5.4 Fail fast if the enabled Stage-1 path cannot construct unambiguous
      bbox groups for the current dataset / batch.
  - v1 behavior is bbox-only
  - non-bbox geometry with `custom.bbox_size_aux.enabled=true` must fail fast

## 6. Validation

- [ ] 6.1 Run targeted config and trainer tests:
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_config_contract.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_objective_atoms_projection.py`
- [ ] 6.2 Add / run Stage-1-focused tests for the new `custom.bbox_size_aux`
      path.
- [ ] 6.3 Add a tiny config-path smoke so at least one authored config resolves
      the new keys without changing decode format or bbox parameterization.

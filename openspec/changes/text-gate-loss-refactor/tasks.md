## 1. Registry + Schema Refactor (Breaking)

- [ ] 1.1 Create a single unified objective/diagnostics module registry (names + allowed config keys + resolution) and delete duplicated per-trainer registries.
- [ ] 1.2 Update Stage-2 pipeline schema to require `stage2_ab.pipeline` and remove flat objective weight knobs from `stage2_ab` (BREAKING).
- [ ] 1.3 Update rollout-matching schema to require `rollout_matching.pipeline` for `stage2_rollout_aligned` and add strict per-module config key allowlists (BREAKING).
- [ ] 1.4 Remove all loss-weight alias keys from schema validation (e.g., `coord_soft_ce_weight`, `coord_w1_weight`, `bbox_*_weight` inside module configs).

## 2. Implement `text_gate` (Correctness-Critical)

- [ ] 2.1 Implement `p_coord(t)` computation from logits via stable logsumexp differences (no new heads); add numeric fences (NaN/Inf clamps, eps).
- [ ] 2.2 Implement `text_gate(t) = -log(1 - p_coord(t) + eps)` and apply it only to supervised `type=struct|desc` positions (exclude `coord`, ignore `eos`).
- [ ] 2.3 Enforce rollout-context masking for gate terms: FP spans excluded; matched-prefix desc excluded where `CE_desc=0`; FN spans included.
- [ ] 2.4 Wire `coord_reg.config.text_gate_weight` into the objective and ensure it contributes to `loss/coord_reg` when non-zero.

## 3. Trainer Integration (Remove Duplication)

- [ ] 3.1 Refactor Stage-2 two-channel trainer to compute `token_ce`, `bbox_geo`, and `coord_reg` via the unified module pipeline (remove precomputed module output bypass).
- [ ] 3.2 Refactor rollout-aligned Stage-2 trainer to use the same unified module pipeline + registry (no separate weight resolution codepaths).
- [ ] 3.3 Remove legacy loss metric aliases and ensure only canonical registry-derived loss keys are emitted (BREAKING).
- [ ] 3.4 Implement sparse metric emission for rollout-only monitors (BREAKING):
  - omit `rollout/*` keys on steps where no rollout was executed (do not emit constant `0.0`),
  - omit `time/mask_build_s` when it is a placeholder (`0.0`),
  - remove `loss/ce`, `loss/coord`, `loss/coord_prefix`, `loss/coord_tail` aliases in favor of canonical registry keys.

## 4. YAML Config Migration (Repo-Wide)

- [ ] 4.1 Update canonical Stage-2 configs under `configs/stage2_two_channel/prod/*.yaml` to define explicit `stage2_ab.pipeline` and remove flat objective knobs (BREAKING).
- [ ] 4.2 Update Stage-2 smoke configs under `configs/stage2_two_channel/smoke/*.yaml` to the same pipeline-only contract (BREAKING).
- [ ] 4.3 Update rollout-aligned configs (if present) to define explicit `rollout_matching.pipeline` and remove legacy aux-loss surfaces (BREAKING).

## 5. Tests (Fail-Fast + Mask Semantics)

- [ ] 5.1 Add unit tests for `text_gate` math: high `p_coord` at text positions increases `loss/text_gate`; low `p_coord` decreases it.
- [ ] 5.2 Add unit tests for rollout-context masking: FP spans do not affect `loss/text_gate`, matched-prefix desc excluded when desc weight is 0.
- [ ] 5.3 Add schema tests that verify unknown module config keys fail fast for both Stage-2 and rollout-matching pipelines.
- [ ] 5.4 Run targeted test suite: `conda run -n ms pytest -q tests/test_stage2_ab_config_contract.py tests/test_stage2_two_channel_training.py`.
- [ ] 5.5 Add a log-key regression test for A-only Stage-2: ensure `rollout/precision|recall|f1` and `time/mask_build_s` are absent when no rollout executed.

## 6. Docs + Runbook Updates (Paper-Ready)

- [ ] 6.1 Update `docs/training/METRICS_LOSSES.md` to document canonical loss keys (and removal of legacy alias keys).
- [ ] 6.2 Update `docs/training/STAGE2_RUNBOOK.md` to show pipeline-only Stage-2 configuration examples (A-only/B-only/AB-mixed) including `text_gate_weight`.
- [ ] 6.3 Add a migration note listing removed keys and canonical replacements (one-time breaking change note).

## 7. Smoke Validation (End-to-End)

- [ ] 7.1 Create a minimal smoke config that differs from prod only by runtime knobs (sample limits/max steps/output dirs) and includes non-zero `text_gate_weight`.
- [ ] 7.2 Run a single-GPU smoke: `config=<smoke_yaml> gpus=0 conda run -n ms bash scripts/train.sh` and verify canonical loss keys appear.

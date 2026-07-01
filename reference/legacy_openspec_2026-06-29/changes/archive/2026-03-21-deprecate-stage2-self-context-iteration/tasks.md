## 1. Runtime and Schema Cleanup

- [x] 1.1 Remove iterative Channel-A self-context execution from
      `src/trainers/stage2_two_channel.py` and collapse Channel-A supervision to
      the single GT-anchored forward only.
- [x] 1.2 Reject deprecated Stage-2 self-context knobs in
      `src/config/schema.py`, `src/config/loader.py`, and `src/sft.py`:
      `n_softctx_iter`, `softctx_grad_mode`, `softctx_temperature`,
      `coord_ctx_embed_mode`, `stage2_ab.coord_decode_mode`, and
      `rollout_matching.coord_decode_mode`.
- [x] 1.3 Remove `self_context` from the teacher-forcing runtime contracts in
      `src/trainers/teacher_forcing/contracts.py`,
      `src/trainers/teacher_forcing/modules/token_ce.py`, and
      `src/trainers/teacher_forcing/modules/coord_reg.py`.
- [x] 1.4 Remove or reject self-context-era Channel-A application presets in
      shared preset validation, including the final-pass preset family in
      `src/trainers/teacher_forcing/module_registry.py` and the rollout-aligned
      teacher-forcing surfaces that still advertise those presets.
- [x] 1.4a Remove the dead self-context token CE stabilizer knob
      `token_ce.config.struct_ce_weight` from shared validation and default
      manifest materialization.
- [x] 1.5 Collapse all Channel-A metric groups from iterative `A*`
      provenance into normal single-pass groups, including `loss/text/*`,
      `loss/coord/*`, `coord_diag/*`, and gradmon coord term names from
      `src/trainers/monitoring/loss_gradient_monitor.py`.

## 2. Config and Test Cleanup

- [x] 2.1 Scrub checked-in active/training Stage-2 YAMLs that still author
      self-context-era knobs or final-pass presets, including
      `configs/stage2_two_channel/base.yaml`,
      `configs/stage2_two_channel/_shared/objective_tuned.yaml`,
      `configs/stage2_two_channel/prod/*.yaml`, and
      `configs/stage2_two_channel/ablation/*.yaml`.
- [x] 2.2 Audit non-training analysis configs that still reference the
      removed semantics and either migrate them or mark them explicitly
      historical,
      including `configs/analysis/rollout_fn_factor_study/*.yaml`.
- [x] 2.3 Convert existing Stage-2 tests from A2/self-context success coverage
      into fail-fast deprecation coverage plus single-pass Channel-A metric
      expectations, including tests that currently assert `loss/A1_*`,
      `loss/A2_*`, `coord_diag/A1/*`, or final-pass preset names.

## 3. Spec and Doc Deprecation

- [x] 3.1 Update the affected OpenSpec capability deltas in this change for
      `stage2-ab-training`, `teacher-forcing-unified-loss-registry`,
      `trainer-metrics-components`, `teacher-forcing-objective-pipeline`, and
      `rollout-matching-sft`.
- [x] 3.2 Mark stable training docs as deprecated or rewritten around the
      single-pass baseline, especially `docs/training/STAGE2_RUNBOOK.md`,
      `docs/training/STAGE2_DESIGN.md`, and `docs/training/METRICS.md`.
- [x] 3.3 Preserve and reference
      `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`
      as the rationale artifact rather than deleting it.
- [x] 3.4 Update `docs/catalog.yaml` so the deprecated training docs and the
      retained diagnostic evidence are discoverable under the new status.

## 4. Verification

- [x] 4.1 Add or update tests so deprecated authored knobs and presets
      hard-fail in active/training configs with actionable migration
      guidance.
- [x] 4.1a Cover the deprecated decode toggles and
      `token_ce.config.struct_ce_weight` with explicit fail-fast tests.
- [x] 4.2 Verify that Stage-2 two-channel training no longer emits
      `self_context`, `loss/A1_*`, `loss/A2_*`, `coord_diag/A1/*`,
      `coord_diag/A2/*`, `A1_coord/*`, or `A2_coord/*` in active runtime
      logs.
- [x] 4.3 Grep the repo to confirm remaining self-context references are limited
      to archived changes, historical diagnostics, or explicit deprecation
      notes, including the final-pass preset family and the removed `A*`
      metric-group names.

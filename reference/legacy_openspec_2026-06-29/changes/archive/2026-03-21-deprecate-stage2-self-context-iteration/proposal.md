## Why

`self-context iteration` is no longer a credible supported training path for
Stage-2 detection work. The current diagnostic record already shows that later
Channel-A self-context passes remain parse-valid while still degrading the
object-level geometry we care about:

- `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`
  records recurrent non-atomic geometry failure modes such as large
  semantic-region boxes, duplicate-heavy narrow strips, and crowd-scene
  collapse even when `sample_valid_pred_rate = 1.0`.
- The repo still treats the concept as an active contract surface through
  `src/config/schema.py` (`Stage2ABConfig`),
  `src/trainers/stage2_two_channel.py`,
  `src/trainers/teacher_forcing/contracts.py`,
  `src/trainers/teacher_forcing/modules/token_ce.py`, and
  `src/trainers/teacher_forcing/modules/coord_reg.py`.
- Stable docs and mainline OpenSpec specs still describe iterative
  self-context, final-pass Channel-A routing, and A2/self-context diagnostics
  as normative behavior, which now conflicts with the intended project
  direction.

We need a cleanup/removal change that makes the invalidated feature impossible
to author accidentally, collapses the runtime back to the supported single-pass
Channel-A path, and marks the stable docs as deprecated instead of silently
rewriting history.

## What Changes

This change deprecates and removes the Stage-2 self-context iteration surface
repo-wide.

- Remove Channel-A iterative self-context execution from the supported
  Stage-2 two-channel contract. Channel-A becomes a single GT-anchored
  teacher-forced forward.
- Reject deprecated authored knobs and concepts that exist only for the
  invalidated loop:
  - `stage2_ab.n_softctx_iter`
  - `stage2_ab.softctx_grad_mode`
  - `stage2_ab.softctx_temperature`
  - `stage2_ab.coord_ctx_embed_mode`
  - `stage2_ab.coord_decode_mode`
  - `rollout_matching.coord_decode_mode`
  - `token_ce.config.struct_ce_weight`
- Reject final-pass/self-context routing presets and migrate the canonical
  teacher-forcing preset guidance to the already-existing single-pass presets:
  - `token_ce.application.preset: anchor_text_only`
  - `bbox_geo.application.preset: anchor_only`
  - `bbox_size_aux.application.preset: anchor_only`
  - `coord_reg.application.preset: anchor_only`
- Remove `self_context` and final-pass runtime/metrics semantics from the
  Stage-2 two-channel training contract and unified loss registry contract.
- Collapse all Channel-A metric groups from the iterative
  provenance-split form into normal single-pass groups:
  - `loss/text/*` instead of `loss/A1_text/*` / `loss/A2_text/*`
  - `loss/coord/*` instead of `loss/A1_coord/*` / `loss/A2_coord/*`
  - `coord_diag/*` instead of `coord_diag/A1/*` / `coord_diag/A2/*`
  - `gradmon/*/coord/*` instead of `gradmon/*/A1_coord/*` /
    `gradmon/*/A2_coord/*`
- Apply the preset cleanup repo-wide across the shared teacher-forcing preset
  surface, including rollout-aligned Stage-2 specs, rather than leaving
  self-context-era preset names discoverable outside `stage2_two_channel`.
- Update OpenSpec and stable training docs so they explicitly record the
  deprecation/removal decision, while preserving historical evidence in
  `progress/diagnostics/`.

## Impact

- Stage-2 becomes easier to reason about and safer to reproduce: one Channel-A
  forward surface, no silent final-pass routing split, no self-context-only
  registry context, and no `A*` Channel-A provenance carryover from the removed loop.
- Active/training configs that still depend on self-context iteration will
  fail fast with migration guidance instead of silently collapsing to a
  different behavior.
- Checked-in active/training YAMLs, docs, and metric expectations that
  still advertise `A*` Channel-A provenance, `self_context`, or final-pass
  preset names must be updated as part of the cleanup.
- Historical diagnostics remain available as evidence; the change deprecates
  the feature, not the record of why it was removed.

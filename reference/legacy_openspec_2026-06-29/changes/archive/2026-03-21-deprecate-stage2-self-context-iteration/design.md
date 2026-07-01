## Summary

This change chooses a hard deprecation and runtime removal for Stage-2
self-context iteration.

- Recommended approach: remove the iterative Channel-A contract, reject the
  authored loop/preset knobs that imply a final self-context pass, and rewrite
  the stable docs/specs around the single-pass Channel-A baseline.
- Historical evidence stays in `progress/diagnostics/`; stable docs are marked
  deprecated or rewritten with a deprecation note rather than erased.
- Geometry decode also collapses to the fixed expectation-decode path used by
  the supported single-pass baseline:
  - remove `stage2_ab.coord_decode_mode`
  - remove `rollout_matching.coord_decode_mode`
  - remove `coord_ctx_embed_mode` and other loop-only knobs
- All Channel-A metrics and diagnostics also collapse back to normal
  single-pass groups instead of retaining iterative provenance tokens:
  - `loss/text/*` replaces `loss/A1_text/*` / `loss/A2_text/*`
  - `loss/coord/*` replaces `loss/A1_coord/*` / `loss/A2_coord/*`
  - `coord_diag/*` replaces `coord_diag/A1/*` / `coord_diag/A2/*`
  - gradmon coord terms use `coord/*` rather than `A1_coord/*` /
    `A2_coord/*`

## Options Considered

### Option 1: Hard remove plus fail-fast on old knobs and presets

Pros:

- makes the invalidated feature impossible to author accidentally
- keeps future experiments from silently inheriting an unsupported path
- aligns code, config, docs, and metrics around one Channel-A contract

Cons:

- breaks authored YAMLs that still mention self-context-era knobs/presets
- requires deliberate doc/spec cleanup across several files

Recommendation: use this option.

### Option 2: Silently coerce old configs to single-pass behavior

Pros:

- fewer immediate breakages for stale YAMLs

Cons:

- hides behavior changes behind the same config surface
- preserves misleading names like `n_softctx_iter` and `self_context`
- makes old experiment definitions harder to audit

Rejected because silent coercion is unsafe for reproducibility.

### Option 3: Doc-only deprecation, keep runtime support

Pros:

- lowest short-term implementation cost

Cons:

- leaves unsupported behavior live in schema, trainer code, and metrics
- keeps stale knobs/presets discoverable
- guarantees future confusion between “historically studied” and “currently
  supported”

Rejected because it does not actually remove the invalidated concept.

## Target State

### Runtime contract

- `custom.trainer_variant: stage2_two_channel` uses a single GT-anchored
  Channel-A teacher-forced forward.
- Channel-A no longer has a final-pass `self_context` surface.
- Channel-A text objective atoms emit only under `loss/text/*`.
- Channel-A coord/bbox objective atoms emit only under `loss/coord/*`.
- Stage-2 coord diagnostics expose `coord_diag/*` for Channel-A and
  `coord_diag/B/*` for Channel-B, with no `coord_diag/A1/*` or
  `coord_diag/A2/*`.
- Loss-gradient monitor coord terms use `coord/*` for Channel-A and
  `B_coord/*` for Channel-B, with no `A1_coord/*` or `A2_coord/*`.

### Deprecated authored surfaces

The following authored surfaces become unsupported in active/training
configs and MUST fail fast with migration guidance:

- `stage2_ab.n_softctx_iter`
- `stage2_ab.softctx_grad_mode`
- `stage2_ab.softctx_temperature`
- `stage2_ab.coord_ctx_embed_mode`
- `stage2_ab.coord_decode_mode`
- `rollout_matching.coord_decode_mode`
- `token_ce.config.struct_ce_weight`
- `token_ce.application.preset: anchor_text_plus_final_struct`
- `bbox_geo.application.preset: anchor_if_single_iter_else_final`
- `bbox_geo.application.preset: final_only`
- `bbox_geo.application.preset: anchor_and_final`
- the equivalent `bbox_size_aux` and `coord_reg` final-pass presets

The intended replacements are:

- `token_ce.application.preset: anchor_text_only`
- `bbox_geo.application.preset: anchor_only`
- `bbox_size_aux.application.preset: anchor_only`
- `coord_reg.application.preset: anchor_only`

## Concrete Handles

Primary runtime surfaces to update:

- `src/config/schema.py`
- `src/config/loader.py`
- `src/sft.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/monitoring/loss_gradient_monitor.py`
- `src/trainers/teacher_forcing/contracts.py`
- `src/trainers/teacher_forcing/modules/token_ce.py`
- `src/trainers/teacher_forcing/modules/bbox_geo.py`
- `src/trainers/teacher_forcing/modules/bbox_size_aux.py`
- `src/trainers/teacher_forcing/modules/coord_reg.py`
- `src/trainers/teacher_forcing/module_registry.py`

Checked-in config and doc surfaces to clean:

- `configs/stage2_two_channel/base.yaml`
- `configs/stage2_two_channel/_shared/objective_tuned.yaml`
- `configs/stage2_two_channel/prod/a_only.yaml`
- `configs/stage2_two_channel/prod/ab_mixed.yaml`
- `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority.yaml`
- `configs/stage2_two_channel/ablation/a_only_iter1.yaml`
- `configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml`
- `configs/stage2_two_channel/ablation/a_only_iter2-res_1024.yaml`
- `docs/training/STAGE2_RUNBOOK.md`
- `docs/training/STAGE2_DESIGN.md`
- `docs/training/METRICS.md`
- `docs/catalog.yaml`

Evidence artifact to retain and reference:

- `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`

## Risks / Trade-offs

- Shared preset validation may still advertise final-pass preset names outside
  Stage-2 two-channel. The implementation should audit both the shared runtime
  registry and the repo-owned rollout-aligned specs so the preset family is not
  still documented as supported anywhere else.
- Some historical analysis configs may mention self-context-era semantics even
  if they no longer run. The cleanup should preserve historical records but
  prevent those configs from being presented as current best practice.
- Tests that assert `loss/A1_*`, `loss/A2_*`, `coord_diag/A1/*`,
  `coord_diag/A2/*`, or self-context-era preset names will need to be
  converted into deprecation or fail-fast coverage.
- The dead `token_ce.config.struct_ce_weight` surface must be removed from both
  authored YAMLs and default pipeline materialization so active configs do not
  carry an inert self-context stabilizer knob.

## Verification

- Config load tests reject the deprecated knobs and final-pass presets with
  actionable migration errors.
- Config load tests reject the deprecated decode toggles and
  `token_ce.config.struct_ce_weight` with actionable migration errors.
- Stage-2 two-channel runtime logs no `self_context`, `loss/A1_*`,
  `loss/A2_*`, `coord_diag/A1/*`, `coord_diag/A2/*`, `A1_coord/*`, or
  `A2_coord/*` keys.
- Stable docs no longer describe self-context iteration as supported behavior
  and instead point readers to the deprecation rationale.
- `rg -n "n_softctx_iter|softctx_grad_mode|softctx_temperature|coord_ctx_embed_mode|self_context|loss/A1_|loss/A2_|coord_diag/A1|coord_diag/A2|A1_coord|A2_coord|anchor_text_plus_final_struct|anchor_if_single_iter_else_final|final_only|anchor_and_final"` only matches
  historical diagnostics, archived changes, or explicit deprecation notes where
  expected.

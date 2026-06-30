# Review Triage

Date: 2026-06-29

Scope: read-only review of the OpenSpec baseline for
`rebuild-coordexp-swift-training-infra`.

Reviewed artifacts:

- `openspec/config.yaml`
- `proposal.md`
- `design.md`
- `tasks.md`
- `specs/**/spec.md`
- `docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md`
- `docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md`
- `agent_verdict/coordexp-swift-codex.md`
- `agent_verdict/coordexp-swift-claude.md`

Review lanes:

- Contract/spec auditor
- Qwen/upstream tracer
- Training/loss/runtime auditor
- Artifact/config/smoke auditor

## P0

No P0 findings were reported.

## P1 Accepted And Patched

- Pre-backward scalar non-finite handling now requires an all-rank reduced
  decision before any rank calls backward, and unsafe scalar state skips
  backward plus optimizer update on all ranks.
- Effective-batch derivation now fails fast for non-divisible effective batch,
  effective batch smaller than world size, backend accumulation conflicts, and
  incomplete final optimizer-step windows.
- The five-step smoke schedule is aligned across docs and OpenSpec: two
  scheduled `eval.forward` runs, normally explicit `eval.forward.steps: [2, 4]`,
  plus the required final checkpoint at planned step 5. The canonical
  `resolved_step_schedule.json` artifact and event fields are now specified.
- Follow-up audit found `tasks.md` still reported completed planning/review
  work as pending. The completed OpenSpec draft, validation, and read-only
  review tasks are now marked complete while source-study, implementation, and
  smoke tasks remain pending.
- Follow-up audit found no-resize image compatibility under-specified. The
  data/template/encoding spec now requires processor-derived admissible
  dimensions, explicit raw-pixel and merged-visual-token caps, and typed
  pre-processor/pre-forward validation diagnostics.

## P2 Accepted And Patched

- Cadence config fields, forbidden aliases, `resolved_step_schedule.json`,
  event lists, fractional milestone rules, dedupe semantics, and minimum event
  fields were promoted into the config/runtime spec.
- Smoke fixture source pinning now covers the preferred exactly-two-object
  source-row path, not only reduced larger rows.
- Earlier non-local cross-review claims in `BLUEPRINT.md` were reworded as
  non-authoritative context; local verdict files remain the auditable evidence.
- Qwen3-VL MRoPE row-shape expectations were made explicit in the packing and
  forward spec.
- Placeholder/grid validation now includes the expanded image-token formula
  based on `image_grid_thw` and `merge_size`.
- DeepSpeed first-smoke status now uses the canonical label vocabulary and
  forbids premature `systems_smoke_verified` or `production_supported`.
- Best-checkpoint selection now excludes update-skipped or unsafe non-finite
  checkpoints by default.
- Resolved-config provenance and run-manifest path/identity requirements were
  strengthened.
- Loss-continuity language now explicitly prevents V1
  `BaseTokenCE + TokenTypeGateLoss` from being interpreted as old production
  coordinate soft-CE or object/role-balanced objective parity.
- Artifact contracts now define minimal required key sets for the run
  manifest, metric events, eval-forward summaries, and checkpoint metadata.

## Wrong Accepted And Patched

- The active data/template spec no longer supports legacy `sorted` object
  ordering. V1 supports `source_order` and deterministic `random`; future
  geometric sorting is reserved under a deliberate name such as
  `geometry_sorted`. The implementation task wording now matches this
  requirement.
- Smoke source wording now uses the current `len12000` source-family naming,
  not the older max-length spelling.

## Duplicate

No material duplicate requirements were reported.

## Stop State

The OpenSpec baseline has no unresolved P0 or P1 review findings after the
accepted patches. This change is ready for user approval as a planning/spec
baseline. It is not approved for source implementation.

Remaining implementation gates:

- dLoRA source study and round-trip probe;
- special-token embedding mechanism source study;
- Qwen3-VL processor, MRoPE, no-resize, placeholder/grid, and FlashAttention
  source probes;
- DeepSpeed systems smoke before production support;
- explicit user authorization before source implementation begins.

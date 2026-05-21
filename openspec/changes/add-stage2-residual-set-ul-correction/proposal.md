## Why

Current Stage-2 rollout-aware training still carries legacy edited-anchor,
false-negative insertion, duplicate-control, and coordinate-repair assumptions.
Those assumptions are no longer the desired research contract.

The new objective should train on the model's own self-prefix states while
keeping the expert target simple:

```text
given a self-prefix:
    compute the remaining supervision set
    supervise the next valid template/action token
    keep schema/text/coord type exclusivity strict
```

The design must support dirty self-prefix recovery, strict token/logit
alignment, K-rollout unlabeled-object (UL) mining, and future ablations without
reviving removed bbox/geometry losses or silently changing baseline objectives.

## What Changes

- Reuse and rewrite this change as the canonical Stage-2 residual-set
  self-prefix correction contract; do not create a second overlapping OpenSpec
  change.
- Define Stage-2 v1 as **round-based offline DAgger-like** training:
  rollout generation is prepared into JSONL first, and training consumes the
  fixed rollout attempts.
- Require the residual-set objective config to name the prepared input at
  `residual_set_correction.config.prepared_rollout_jsonl`.
- Treat K rollout attempts as equal self-prefix samples, with exact duplicate
  attempt deduplication only.
- Replace anchor/explorer terminology in the new path with `rollout_attempt`
  and `rollout_id`; legacy adapters may translate old fields offline only.
- Add a shared `SupervisionAtom` / target IR contract using canonical
  `logit_position`; v1 maps onto the current shared IR where `target_position`
  and `selected_token_id` are required validator fields.
- Require Stage-2 target assembly to go through a Stage-1-compatible
  `TemplateBoundaryAdapter` rather than hand-authored schema string
  concatenation.
- Enable standalone global token-type exclusivity by default for
  schema(struct)/text(desc)/coordinate stability; malformed segments are dropped
  or masked/resynced rather than weakening schema supervision.
- Define the inner loss as valid-set marginal likelihood over valid next-token
  ids derived from transition-validated `ValidAction` records; hard CE and EOS
  are singleton valid-set cases.
- Use one rollout attempt as one training sequence with all eligible correction
  atoms active, normalized by per-sequence weighted mean.
- Keep dirty prefix tokens as masked context when reliable resynchronization is
  possible, while never training malformed span tokens directly.
- Delete the earlier `bbox_tail_from_anchor` / coordinate repair plan from
  Stage-2 v1. Bbox geometry is a commitment gate, not a repair target.
- Mine promoted UL objects from strict cross-rollout consensus, with reviewable
  artifacts, default UL weight `0.5`, no hard per-sample cap, rollout-local
  member bbox supervision, and a near-GT gray zone to avoid promoting
  localization bias.
- Treat high-IoU wrong-description conflicts as low-weight label-conflict
  corrections, not as TP, not as UL, and not as emitted objects.
- Preserve existing hard-SFT/current Stage-2 objective paths as explicit
  baselines; the new behavior is opt-in through the residual-set objective.

## Capabilities

### Added Capabilities

- `stage2-residual-set-correction`: Defines offline rollout-attempt input,
  exact deduplication, residual-set scan state, dirty-prefix recovery,
  correction atoms, token/logit alignment, type loss, valid-set marginal loss,
  UL promotion, label-conflict handling, artifacts, and diagnostics.

### Modified Capabilities

- `stage2-ab-training`: Adds the strict residual-set objective path and its
  prepared-rollout JSONL, module/config, artifact, and metric contracts while
  keeping legacy baselines selectable.
- `teacher-forcing-unified-loss-registry`: Adds residual-set valid-set marginal,
  token-type exclusivity, atom weighting/normalization, and provenance metrics
  without restoring removed geometry or duplicate losses.
- `teacher-forcing-objective-pipeline`: Requires residual-set Stage-2 builders
  to compile into shared teacher-forcing IR and keeps loss modules ignorant of
  FN/FP/duplicate/UL mining semantics.

## Impact

- Affected training surfaces:
  - `configs/stage2_two_channel/**`
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/**`
  - `src/trainers/teacher_forcing/**`
  - `src/training/teacher_forcing/**`
  - `src/detection/template.py`
  - `src/detection/tokenization.py`
  - `src/common/detection_sequence.py`
  - `src/metrics/**`
- Affected stable docs/specs:
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
  - `docs/IMPLEMENTATION_MAP.md`
  - `openspec/specs/stage2-ab-training/spec.md`
  - `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
  - `openspec/specs/teacher-forcing-objective-pipeline/spec.md`
  - new or updated `openspec/specs/stage2-residual-set-correction/spec.md`
- Affected artifacts and diagnostics:
  - prepared rollout JSONL records with `response_token_ids`;
  - `monitor_dumps/step_*.json` counters for clean/dirty correction, malformed
    recovery, type diagnostics, UL mining, and label conflicts;
  - `monitor_dumps/ul_clusters.jsonl` for promoted/rejected/review UL clusters.

## Non-Goals

- No full recursive autoregressive subtree marginal over alternative hidden
  states.
- No online generate-while-training loop in v1.
- No duplicate-burst unlikelihood, coord regularizer, bbox geometry auxiliary,
  bbox size auxiliary, coordinate repair, nearest-GT repair, or regression-ish
  bbox objective.
- No default clean GT SFT mixing; it may exist as an explicit stabilizer with
  default mix `0`.
- No automatic mutation of dataset GT from promoted UL mining.
- No default PNG visualization generation from training.

## Verification Scope

This OpenSpec change defines the stable contract. Smoke or val200 performance
improvement is not an OpenSpec validity gate. Implementation must still provide
targeted tests and smoke runs for:

- prepared-rollout replay and token/logit alignment;
- type loss and valid-set marginal behavior;
- dirty-prefix resync/drop behavior;
- UL promotion/rejection/gray-zone behavior;
- Stage-2 runnable overfit/smoke sanity on prepared rollouts;
- baseline configs remaining selectable and unmutated.

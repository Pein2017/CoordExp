# P0 Stage-2 Eval Validity Implementation Plan

> **Scope:** Implement only the approved P0 metric-bearing Stage-2 eval-validity
> hardening slice. Do not move Stage-2 runtime projection, target construction,
> DDP/packing ownership, A/B cleanup, or broader shared-inference adapters in
> this plan.

**Goal:** Stage-2 metric-bearing eval must fail before official artifact
materialization when exact source image identity/geometry, strict parser status,
or comparable prompt/decode/model/score provenance is missing.

**Worktree:** `/data/CoordExp/.worktrees/harden-unified-training-runtime-boundaries`

**OpenSpec Change:** `harden-unified-training-runtime-boundaries`

**Validation Command:** `openspec validate harden-unified-training-runtime-boundaries --strict`

`openspec` is not currently available on this shell PATH, so the implementation
must still name the validation command and report it as skipped unless the CLI
becomes available before completion.

## Current Evidence

Read-only subagent review converged on the same P0 seams:

- `src/trainers/rollout_aligned_evaluator.py` fabricates `image_<idx>.jpg` and
  defaults missing dimensions to `1000x1000` inside
  `build_eval_detection_record_confidence_postop_input`.
- `_materialize_stage2_eval_artifacts` writes `gt_vs_pred.jsonl`,
  `gt_vs_pred_scored.jsonl`, and provenance sidecars before any row-level
  official validity check.
- Stage-2 eval parser adapters can consume salvage/diagnostic output before
  metric-bearing artifacts are written.
- Existing comparable-artifact provenance checks reject missing score
  fingerprints and `metric_bearing=false`, but Stage-2 eval does not expose or
  enforce strict parser status before official metric materialization.

## Files In Scope

Production:

- `src/trainers/rollout_aligned_evaluator.py`
- `src/trainers/stage2_rollout_runtime.py`
- `src/infer/artifacts.py`
- `src/infer/parsing.py`
- `src/training/stage2/rollout_codec.py`

Tests:

- `tests/test_stage2_rollout_runtime.py`
- `tests/test_parser_policy_parity.py`
- `tests/test_score_policy_fingerprint.py` if comparable provenance enforcement
  needs a score-fingerprint-level test.

Docs/spec task tracking only:

- `openspec/changes/harden-unified-training-runtime-boundaries/tasks.md`
- this implementation plan

No config migration, dataset rewrite, or stable docs-route rewrite is in this
P0 implementation plan.

## Acceptance Criteria

- Stage-2 official eval raises before creating `eval_detection/step_<step>/`
  official artifacts if a metric-bearing row lacks a real `image`/`images`
  value.
- Stage-2 official eval raises before artifact writes if `width` or `height` is
  missing, non-integer, or non-positive.
- Stage-2 official eval raises before artifact writes if parser status is
  diagnostic, salvage-recovered, fallback-based, or otherwise
  `metric_bearing=false`.
- `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`,
  `gt_vs_pred_scored.jsonl.provenance.json`, and `raw_rollouts.jsonl` are not
  partially written on P0 validation failures.
- Stage-2 official eval raises before artifact writes if metric-bearing
  rollout rows lack non-empty backend-recorded `prompt_token_ids`.
- Stage-2 official eval raises before artifact writes if backend-recorded
  `prompt_token_ids` contain non-integer elements such as strings or floats.
- Stage-2 official eval rows carry and validate a stable source record identity
  (`source_record_id`, plus available `sample_id`/`base_idx`/`image_id`).
- Stage-2 official eval rejects multi-image inputs rather than collapsing visual
  provenance to the first image.
- Stage-2 score sidecar prompt provenance is derived from the materialized
  rollout artifacts: prompt-token hashes, detection sequence format,
  object-order policy, rollout template summary, and validated visual source
  metadata. It must not claim offline/online prompt parity without verified
  prompt-token and visual metadata parity.
- Stage-2 score sidecar prompt fingerprint changes when the recorded prompt
  token surface or detection sequence format changes.
- Positive Stage-2 eval tests still materialize the existing artifact names and
  can be loaded via `load_comparable_artifact(..., require_score=True)`.
- Broader P1/P2 architecture tasks remain unchecked and unimplemented.

## TDD Tasks

1. Add failing Stage-2 eval tests for missing source image identity:
   - Build an eval sample with `width`/`height` but no `images` or `image`.
   - Assert `trainer.evaluate()` raises with a message naming metric-bearing
     source image identity.
   - Assert `eval_detection/step_0000011` does not exist.

2. Add failing Stage-2 eval tests for missing source dimensions:
   - Build an eval sample with real image identity but missing `width` or
     `height`.
   - Assert failure before official artifact materialization.

3. Add failing Stage-2 eval tests for parser salvage/diagnostic status:
   - Use the compact-full parser path with malformed neighboring output that
     currently salvages one valid row.
   - Assert official eval rejects the row before `gt_vs_pred*.jsonl` is written.
   - Prefer a focused helper/unit test if a full trainer path would require
     fragile token fixtures.

4. Add or extend parser-policy tests:
   - Stage-2 parser adapters must expose metric-bearing metadata.
   - Diagnostic/salvage parser results must fail `require_metric_bearing` with a
     consumer-specific message.

5. Implement minimal code:
   - Remove metric-bearing fallback fabrication from
     `build_eval_detection_record_confidence_postop_input`; missing identity or
     dimensions should raise a clear `ValueError`.
   - Add parser-status metadata to Stage-2 parsed rollout results.
   - Validate parser status before appending official base/scored records.
   - Validate gathered rows in `_materialize_stage2_eval_artifacts` before
     creating/writing official files, so partial artifacts are not emitted.
   - Tighten score provenance only as much as needed for strict parser status
     and real prompt-token/visual-source binding; avoid broader provenance
     schema redesign.

6. Verification:
   - Run targeted failing tests before implementation where practical.
   - Run targeted passing tests after implementation:
     `conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py tests/test_parser_policy_parity.py tests/test_score_policy_fingerprint.py`
   - If `conda` is unavailable, try `python -m pytest` for the same targets.
   - Report skipped OpenSpec CLI validation if `openspec` remains unavailable.

## Implementation Notes

- Source identity and dimensions are now validated before official artifact
  directory creation; missing/fabricated images and missing/non-exact dimensions
  produce invalid gathered payloads and fail on rank 0 without DDP early-exit
  deadlock.
- Parser adapters now expose `DetectionParserResult` metadata; diagnostic,
  salvage, fallback, truncated, invalid, dropped-object, or empty-valid-set
  parser results are not metric-bearing for official Stage-2 eval.
- Compact-full salvage remains available for diagnostic/training-side
  inspection, but it is explicitly marked `fallback_reason=compact_full_salvage`
  and rejected from metric-bearing official eval.
- Score sidecars now include `prompt_provenance` and fingerprint the
  backend-recorded prompt-token surface plus the Stage-2 prompt/template/visual
  policy summary. Missing prompt-token metadata fails before writing
  `eval_detection/step_<step>/`.
- Score sidecars now require `eval_rollout_artifacts_all` even when callers
  provide precomputed prompt/parser provenance maps; direct no-artifact writer
  calls raise before writing a comparable sidecar.
- Parser exceptions in official eval are converted into gathered invalid
  artifacts so DDP ranks still reach the gather/broadcast failure path.
- Metric-bearing base/scored rows now carry `source_record_id`; multi-image
  official eval inputs are rejected explicitly as out of scope for this P0
  single-image surface.
- Current focused verification command:
  `/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py tests/test_parser_policy_parity.py tests/test_score_policy_fingerprint.py`
  passed with `134 passed, 2 warnings` on 2026-05-27.
- OpenSpec CLI validation is still blocked in this shell by
  `openspec: command not found`; command remains
  `openspec validate harden-unified-training-runtime-boundaries --strict`.

## Review Gates

After implementation:

- Spec compliance review must verify every P0 acceptance criterion and confirm
  no broader runtime-boundary work was implemented.
- Code quality review must check the new validation helpers are small, pure
  where possible, and do not introduce DDP deadlock risk or partial writes.
- Final verification must include targeted pytest output or a concrete skipped
  reason.

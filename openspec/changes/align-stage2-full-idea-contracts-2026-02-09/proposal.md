# Change: Align Stage-2 AB + Data Contract with Updated Full Idea

## Why
`progress/full_idea.md` was updated with stricter Stage-2 semantics (Unified Channel-B, FP-neutral policy, stop-neutral CE, and FN injection/key-allocation requirements), but current implementation/docs/specs are partially misaligned.

Key drifts that now affect correctness and reproducibility:
- Channel-B CE masking currently under-supervises rollout prefix structure in the non-reordered path.
- FN key numbering is derived from valid parsed objects, not the full set of `object_N` keys present in the retained prefix body.
- Stage-2 defaults/config docs still emphasize `reordered_gt_sft` behavior as the normal path.
- Dataset JSON builder can emit geometry-less objects, violating the declared object contract.

Without a contract-first refactor, implementation and governance artifacts will continue to diverge and make Stage-2 results hard to interpret or reproduce.

## OpenSpec Workflow (Migration-Aware)
- This change uses the `spec-driven` schema (`.openspec.yaml`).
- Under this schema, implementation is gated by artifacts in order:
  - `proposal` -> `design` + `specs` -> `tasks` -> code implementation.
- Historical status at proposal drafting time (2026-02-09; this snapshot may become stale) was:
  - `proposal`: done
  - `design`: ready
  - `specs`: ready
  - `tasks`: blocked (requires `design` and `specs`)
- For current status, use:
  - `openspec status --change align-stage2-full-idea-contracts-2026-02-09 --json`
- This proposal intentionally precedes tasks/code; tasks are created after design/spec deltas are written.

## What Changes
- Refine Stage-2 AB Channel-B contract to match updated `full_idea` defaults:
  - Unified one-pass teacher-forced update over rollout-prefix + FN-injected entries.
  - CE policy explicitly enforces:
    - matched prefix structure supervision,
    - matched prefix desc masking,
    - FP full-mask (CE/geometry neutral),
    - FN-injected struct+desc supervision,
    - stop-neutral masking on outermost `}` and `<|im_end|>`.
  - Geometry policy explicitly enforces:
    - `L_geo_matched` on matched prefix objects,
    - `L_geo_FN` on FN-injected objects,
    - FP contributes no geometry loss.
- Formalize FN key allocation/injection rule:
  - derive `start_id` from all `object_N` keys in the retained prefix body (not only valid matched objects),
  - locate the outermost top-level close brace via brace-depth scan (no "last `}` token" heuristic),
  - inject FN entries before that same outermost close brace,
  - insert a comma before the first FN entry iff the retained prefix body already contains at least one object entry,
  - use the same deterministic outermost-brace target for both FN injection and stop-neutral CE masking.
- Rebaseline Stage-2 defaults to the updated idea:
  - make Unified Channel-B behavior the default,
  - keep `reordered_gt_sft` as legacy/experimental (opt-in ablation), not default training behavior.
- Tighten dataset runtime contract enforcement for assistant payload construction:
  - each emitted object must have non-empty `desc` and exactly one geometry field (`bbox_2d` xor `poly`),
  - enforce geometry arity invariants:
    - `bbox_2d` length exactly 4,
    - `poly` flattened length even and >= 6,
  - reject malformed/missing/ambiguous geometry at runtime (fail-fast) instead of serializing invalid objects.
- Sync OpenSpec specs and runbook/metrics docs to the new ground truth.

Non-goals (this change):
- No new CLI flags.
- No architecture forks or new detector heads.
- No changes to upstream HF model files.

## Capabilities
- `stage2-ab-training` (modified): Unified Channel-B CE/geometry semantics, deterministic FN injection, stop-neutral brace targeting.
- `rollout-matching-sft` (modified, scoped clarifications only): prefix key-accounting and append/injection determinism where wording is ambiguous.
- `public-data-pipeline` (modified, if required by ownership boundary): strict runtime object geometry invariants for emitted assistant payloads.

## Impact
Affected specs:
- `openspec/specs/stage2-ab-training/spec.md`
- `openspec/specs/rollout-matching-sft/spec.md` (clarification-only if needed; key-allocation drift is primarily implementation)
- (If needed for strict dataset-runtime guarantees) `openspec/specs/public-data-pipeline/spec.md`

Affected code (expected):
- `src/trainers/stage2_ab_training.py` (Channel-B CE/span policy, key allocation, meta contracts)
- `src/trainers/rollout_matching_sft.py` (prefix key accounting / injection helpers)
- `src/datasets/builders/jsonlines.py` (object emission validation)
- `src/datasets/utils.py` (strict geometry-field/arity extraction contract)
- `configs/stage2_ab/**` (default behavior migration)
- `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS_LOSSES.md`, `docs/data/JSONL_CONTRACT.md` (contract alignment)

Validation impact:
- Update/add unit tests for:
  - non-reordered Channel-B CE span policy:
    - matched prefix structure supervised, matched desc masked, FP fully masked, FN struct+desc supervised.
  - FN key collision-safe numbering:
    - `start_id` derived from all retained prefix `object_N` keys, including invalid/dropped objects.
  - deterministic brace-depth + comma insertion behavior:
    - FN injected before outermost close brace; comma inserted only when prefix body already has object entries.
  - stop-neutral marker handling:
    - CE mask on the same outermost close brace used by injection and on `<|im_end|>`.
  - dataset builder geometry invariants:
    - reject desc-only objects, reject multi-geometry objects, enforce bbox/poly arity rules.

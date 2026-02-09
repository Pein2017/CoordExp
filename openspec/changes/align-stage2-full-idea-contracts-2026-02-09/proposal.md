# Change: Align Stage-2 AB + Data Contract with Updated Full Idea

## Why
`progress/full_idea.md` was updated with stricter Stage-2 semantics (Unified Channel-B, FP-neutral policy, stop-neutral CE, and FN injection/key-allocation requirements), but current implementation/docs/specs are partially misaligned.

Key drifts that now affect correctness and reproducibility:
- Channel-B CE masking currently under-supervises rollout prefix structure in the non-reordered path.
- FN key numbering is derived from valid parsed objects, not the full set of `object_N` keys present in the retained prefix body.
- Stage-2 defaults/config docs still emphasize `reordered_gt_sft` behavior as the normal path.
- Dataset JSON builder can emit geometry-less objects, violating the declared object contract.

Without a contract-first refactor, implementation and governance artifacts will continue to diverge and make Stage-2 results hard to interpret or reproduce.

## What Changes
- Refine Stage-2 AB Channel-B contract to match updated `full_idea` defaults:
  - Unified one-pass teacher-forced update over rollout-prefix + FN-injected entries.
  - CE policy explicitly enforces:
    - matched prefix structure supervision,
    - matched prefix desc masking,
    - FP full-mask (CE/geometry neutral),
    - FN-injected struct+desc supervision,
    - stop-neutral masking on outermost `}` and `<|im_end|>`.
- Formalize FN key allocation/injection rule:
  - derive `start_id` from all `object_N` keys in the retained prefix body (not only valid matched objects),
  - inject FN entries before the outermost top-level close brace.
- Rebaseline Stage-2 defaults to the updated idea:
  - make Unified Channel-B behavior the default,
  - keep `reordered_gt_sft` as legacy/experimental (opt-in ablation), not default training behavior.
- Tighten dataset runtime contract enforcement for assistant payload construction:
  - each emitted object must have non-empty `desc` and exactly one geometry field,
  - reject malformed geometry combinations/arity instead of serializing invalid objects.
- Sync OpenSpec specs and runbook/metrics docs to the new ground truth.

Non-goals (this change):
- No new CLI flags.
- No architecture forks or new detector heads.
- No changes to upstream HF model files.

## Impact
Affected specs:
- `openspec/specs/stage2-ab-training/spec.md`
- `openspec/specs/rollout-matching-sft/spec.md`
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
  - Channel-B CE span policy,
  - FN key collision-safe numbering,
  - stop-neutral marker handling after FN injection,
  - dataset builder geometry invariants.

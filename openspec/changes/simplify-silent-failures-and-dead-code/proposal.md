## Why

CoordExp currently contains a growing set of “best-effort” blocks that swallow all exceptions (`except Exception: pass`) in core code paths (datasets, trainers, and pipeline glue). This makes failures non-obvious, can leak mutable state (e.g., prompt/template overrides), and undermines reproducibility and paper-ready debugging.

Recent work surfaced how redundant defensive logic can accumulate quickly in hot entrypoints; we need a strict, reviewable simplification policy plus targeted removal of dead code to keep the research stack maintainable.

## What Changes

- Remove dead/unreferenced helper functions in training + inference modules (no intended behavior change).
- Introduce a narrow “silent failure” policy:
  - **BREAKING**: core execution paths MUST NOT swallow unexpected exceptions; failures become fail-fast or explicitly logged.
  - Only explicitly-justified sinks (e.g., log tee I/O) may suppress exceptions.
- Remove legacy/no-op surfaces when safe for the current pipeline (no external dependencies):
  - delete unused legacy prompt aliases,
  - delete deprecated config placeholders that can never be enabled.
- Tighten dataset prompt/metadata handling:
  - Per-sample prompt injection MUST be restored deterministically (no prompt leakage across samples).
  - Encoded samples MUST carry stable join metadata (`sample_id`, `dataset`, `base_idx`) for debugging/mining.
- Align evaluator behavior with simplified deprecation handling:
  - Deprecated legacy keys are removed/rejected (fail-fast) rather than silently ignored.

## Capabilities

### New Capabilities
- `silent-failure-policy`: Define the allowed exception-handling patterns in CoordExp and require explicit fail-fast/logging behavior in core training/infer/eval paths.

### Modified Capabilities
- `fusion-dataset`: Dataset encoding/prompt handling and sample metadata attachment become deterministic and non-silent on failure.
- `detection-evaluator`: Deprecated legacy keys are rejected (fail-fast) and removed from supported config surface (no backward/legacy support, no warnings).

## Impact

- Affected code (non-exhaustive):
  - `src/datasets/dense_caption.py`
  - `src/datasets/unified_fusion_dataset.py`
  - `src/trainers/stage2_ab/scheduler.py`
  - `src/trainers/stage2_ab_training.py`
  - `src/trainers/rollout_matching_sft.py`
  - `src/infer/pipeline.py`
  - `src/eval/detection.py`
  - `src/config/schema.py`
  - `src/config/prompts.py`
- Correctness / reproducibility:
  - Runs that previously masked errors will now stop early or surface actionable logs, improving experiment validity.
- Non-goals:
  - No changes to geometry semantics, packing semantics, or training algorithms.

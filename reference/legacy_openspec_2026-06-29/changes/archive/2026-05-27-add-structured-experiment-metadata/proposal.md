## Why

CoordExp training runs currently overload `training.run_name` and
`training.artifact_subdir` with experiment semantics such as ablation identity,
feature flags, and runtime choices. That makes leaf configs hard to maintain
and forces retrospective analysis to recover experiment intent from brittle
path fragments instead of first-class structured data.

We need a config-first way to record both authored experiment intent and the
executed runtime context so runs are easier to audit, compare, and revisit
without redefining `resolved_config.json` or hiding semantics in naming
conventions.

## What Changes

- Add a top-level `experiment` config section for authored, human-readable run
  context such as purpose, hypothesis, key deviations, runtime notes, and
  optional comments.
- Emit a new `experiment_manifest.json` training artifact that combines:
  authored experiment metadata, run identity, a hard-runtime summary,
  provenance summary, and pointers to the authoritative run artifacts.
- Keep `resolved_config.json`, `effective_runtime.json`, `pipeline_manifest.json`,
  and `run_metadata.json` as concern-specific artifacts instead of collapsing
  everything into one raw dump.
- Update training docs and targeted tests so experiment metadata becomes the
  stable run-level entrypoint for retrospective analysis.
- Preserve compatibility for legacy configs that do not yet author
  `experiment.*` by emitting an empty/null authored block rather than failing
  the run.

## Capabilities

### New Capabilities
- `experiment-metadata`: Structured authored experiment intent and a unified
  run-level experiment manifest for training runs.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `src/config/schema.py`
  - `src/config/loader.py`
  - `src/bootstrap/run_metadata.py`
  - `src/utils/run_manifest.py`
  - `src/sft.py`
  - `src/analysis/unmatched_proposal_verifier.py`
- Affected artifacts:
  - new `experiment_manifest.json`
  - clarified ownership of `run_metadata.json` as low-level provenance
- Affected docs/tests:
  - `docs/ARTIFACTS.md`
  - `docs/SYSTEM_OVERVIEW.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - targeted run-manifest / config-schema tests

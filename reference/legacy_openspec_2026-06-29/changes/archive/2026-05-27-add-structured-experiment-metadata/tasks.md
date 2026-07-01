## 1. Schema And Artifact Design

- [x] 1.1 Add a typed top-level `experiment` training-config section with strict key validation and resolved-config serialization coverage.
- [x] 1.2 Implement `experiment_manifest.json` emission from the training bootstrap path using authored experiment metadata, runtime summary, provenance summary, and artifact pointers.
- [x] 1.3 Keep `run_metadata.json` as the low-level provenance artifact and update repo-owned consumers that search run-level metadata to recognize the new manifest.

## 2. Documentation And Examples

- [x] 2.1 Update `docs/ARTIFACTS.md`, `docs/SYSTEM_OVERVIEW.md`, and `docs/training/STAGE2_RUNBOOK.md` to define the new artifact roles and the `experiment` authoring pattern.
- [x] 2.2 Add at least one representative training config example that authors the new `experiment` section without relying on semantic `run_name` parsing.

## 3. Verification

- [x] 3.1 Add or update targeted tests for experiment schema parsing, experiment-manifest writing, and metadata-consumer fallback behavior.
- [x] 3.2 Run focused verification:
  `conda run -n ms python -m pytest -q tests/test_run_metadata_file.py tests/test_run_manifest_files.py tests/test_training_config_strict_unknown_keys.py`

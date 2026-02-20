## Scope

Audit target: all `try/except` blocks under `src/`.

Guiding policy (strict-by-default):
- unexpected internal exceptions must not be silently swallowed,
- operator-controlled input violations (training inputs and inference/eval inputs) MUST fail fast (raise); no skip-and-continue,
- semantics-changing fallbacks (default returns, empty outputs) are forbidden in core correctness paths; continue-but-observable is allowed only for explicitly salvage-mode training subpaths consuming model-generated outputs,
- remove redundant try/except wrappers that do not add actionable context (“strip over-engineering”).

## Status (current)

As of this change revision, the high-signal swallow/fallback patterns found during the initial audit have been removed,
and the policy is enforced by CI-level tests that block silent exception suppression patterns under `src/`.

### P0
- None remaining in the audited surfaces (core inference + policy scan + major offenders).

### P1
- None remaining: augmentation curriculum parsing/overrides now fail fast on invalid values and unknown ops/params (no silent drift).

### P2
- None remaining: redundant try/except wrappers that did not add actionable context were removed.

## Resolved findings (historical)

- `src/trainers/rollout_matching/parsing.py` — removed blanket suppression around prefix parsing; unexpected exceptions now propagate.
- `src/trainers/rollout_matching/matching.py` — removed blanket exception-to-default-value fallbacks in matching helpers.
- `src/eval/detection.py` — removed blanket exception-to-default-value fallbacks in metric helpers.
- `src/common/prediction_parsing.py` — narrowed exception handling to expected parse/validation failures only.
- `src/datasets/preprocessors/augmentation.py` — removed blanket import guards; curriculum parsing/override application now fails fast on invalid values.
- `src/datasets/preprocessors/resize.py` — removed exception-driven last-resort fallbacks without diagnostics.
- `src/utils/logger.py` — narrowed best-effort I/O exception handling to expected filesystem-related failures and preserved diagnostics.

## Verification

- Policy scan (blocking): `conda run -n ms pytest -q tests/test_silent_failure_policy.py`
- Curriculum fail-fast contract: `conda run -n ms pytest -q tests/test_augmentation_curriculum_contract.py`

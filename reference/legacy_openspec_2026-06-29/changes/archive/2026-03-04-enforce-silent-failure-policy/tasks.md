## 0. Audit Baseline

- [x] 0.1 Inventory and classify `try/except` usage across core `src/` surfaces.
- [x] 0.2 Capture an evidence-backed audit snapshot in `audit_try_except.md`.
- [x] 0.3 Preserve historical resolved offenders in the audit report for traceability.

## 1. Spec Contract Refresh

- [x] 1.1 Rewrite change delta specs into OpenSpec-compliant sections (`## ADDED|MODIFIED Requirements`) with `#### Scenario:` blocks.
- [x] 1.2 Align terminology across specs: operator-controlled input violations vs model-output consumer invalidity vs unexpected internal exceptions.
- [x] 1.3 Keep explicit best-effort sink carve-out language, but constrain it to non-correctness paths.

## 2. Enforcement Artifacts (Current State)

- [x] 2.1 Keep AST-based policy checks in `tests/test_silent_failure_policy.py` for Tier-0/Tier-1 blanket suppression patterns.
- [x] 2.2 Keep scanner regression coverage that proves detection of `pass` / `continue` / default-return patterns.
- [x] 2.3 Restore policy test pass status by eliminating currently flagged blanket suppressions in active Stage-2 trainer paths.

## 3. Code Follow-Up (Remaining)

- [x] 3.1 Classify each currently flagged handler as either:
  - legitimate sink-scoped best-effort (diagnostic-only), or
  - correctness-path suppression requiring fail-fast/narrow handling.
- [x] 3.2 Replace correctness-path blanket suppression (`pass`, `continue`, default-return under blanket `Exception`) with narrow exceptions and fail-fast behavior.
- [x] 3.3 For legitimate sink-scoped best-effort handlers, narrow exception classes where feasible and keep warning/counter observability.
- [x] 3.4 Refresh `audit_try_except.md` after remediation with resolved vs remaining findings and file:line evidence.

## 4. Verification

- [x] 4.1 Run `conda run -n ms python -m pytest -q tests/test_silent_failure_policy.py` and require pass.
- [x] 4.2 Run targeted policy-adjacent checks:
  - `conda run -n ms python -m pytest -q tests/test_no_silent_except_exception_pass.py`
  - `conda run -n ms python -m pytest -q tests/test_batch_extras_failure_not_silent.py`
  - `conda run -n ms python -m pytest -q tests/test_augmentation_curriculum_contract.py`
- [x] 4.3 Run one representative Stage-2 smoke to verify:
  - unexpected exceptions terminate non-zero,
  - permitted sink-scoped best-effort behavior remains observable.

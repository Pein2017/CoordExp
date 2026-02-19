## 0. Audit (inventory + classification)

- [ ] 0.1 Enumerate all `try/except` blocks under `src/` and record:
  - file + symbol/function context,
  - exception types caught (bare / `Exception` / specific),
  - behavior (`raise`, `pass`, `continue`, default-return, log+continue, etc.),
  - whether the handler is in a core execution path.
- [ ] 0.2 Classify each block:
  - **Keep** (narrow + necessary + observable),
  - **Narrow** (broad catch but required; reduce to specific exceptions),
  - **Remove** (redundant wrapper / over-engineering),
  - **Fix** (silent swallow / semantics-changing fallback).
- [ ] 0.3 Produce an evidence-backed audit report under `openspec/changes/enforce-silent-failure-policy/audit_try_except.md` with P0/P1/P2 severity.

## 1. Spec deltas (policy contract)

- [ ] 1.1 Update `silent-failure-policy` spec to expand “silent swallowing” beyond `except ...: pass` to include `continue` / default-return / semantics-changing fallbacks in core paths.
- [ ] 1.2 Update `silent-failure-policy` spec to define strict-by-default behavior and the narrow best-effort I/O sink carve-out.
- [ ] 1.3 Update `inference-engine` spec to clarify expected per-sample error handling vs unexpected internal exceptions (fail fast), and align scenarios accordingly.

## 2. CI enforcement (minimal, high-signal)

- [ ] 2.1 Replace/extend `tests/test_silent_failure_policy.py` with an AST-based scan that fails on:
  - bare `except:` in `src/`,
  - `except Exception` / `except BaseException` in core paths,
  - broad exception handlers that do not re-raise and do not emit structured errors + counters.
- [ ] 2.2 Add targeted regression tests for known offenders (at least one representative for: `pass`, `continue`, default-return).

## 3. Code fixes (fail fast + remove over-engineering)

- [ ] 3.1 Remove blanket suppression in core paths (e.g., `except Exception: pass`) and replace with:
  - explicit exception types, and either
  - re-raise with context, or
  - expected-error recording (in inference/eval only).
- [ ] 3.1.1 Fix known P0 offenders (evidence-backed):
  - `src/trainers/rollout_matching/parsing.py` — replace `except Exception: pass` around `parse_coordjson(...)` with explicit expected parse exception handling; unexpected exceptions must propagate.
  - `src/trainers/rollout_matching/matching.py` — replace `except Exception: return 0.0` in maskIoU with input validation + narrow exceptions + explicit counters (or fail fast on unexpected).
  - `src/eval/detection.py` — replace `except Exception: return 0.0` in `_segm_iou` with validation + narrow exceptions + explicit counters (or fail fast on unexpected).
  - `src/common/prediction_parsing.py` — narrow salvage parsing exception types; ensure unexpected exceptions are not converted into “empty_pred”.
- [ ] 3.1.2 Fix known dataset-side silent fallbacks:
  - `src/datasets/preprocessors/augmentation.py` — remove hidden config drift (`return current` / `continue` on override failure) and make failures observable.
  - `src/datasets/preprocessors/resize.py` — make “filename-only” relativization and EXIF fallback observable; narrow broad exceptions.
- [ ] 3.2 Replace semantics-changing fallbacks (e.g., return `0.0`, empty outputs) with explicit failure or explicit per-sample error records.
- [ ] 3.3 Tighten optional dependency handling to `ImportError` / `ModuleNotFoundError` with actionable guidance (no blanket catch).
- [ ] 3.4 Strip redundant `try/except` wrappers that only re-raise without adding context; keep context at meaningful boundaries.
- [ ] 3.5 Ensure any `finally` blocks that restore temporary mutable state remain deterministic; restoration failures terminate the run.

## 4. Verification

- [ ] 4.1 Run unit tests: `conda run -n ms python -m pytest tests/test_silent_failure_policy.py`.
- [ ] 4.2 Run a targeted smoke on one representative pipeline path affected by changes (training or inference), verifying:
  - unexpected exceptions stop the run,
  - expected per-sample errors are recorded with counters (where permitted),
  - no “quiet” defaults mask failures.

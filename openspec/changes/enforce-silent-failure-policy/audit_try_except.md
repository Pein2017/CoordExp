## Scope

Audit target: all `try/except` blocks under `src/`.

Guiding policy (strict-by-default):
- unexpected internal exceptions must not be silently swallowed,
- operator-controlled input violations (training inputs and inference/eval inputs) MUST fail fast (raise); no skip-and-continue,
- semantics-changing fallbacks (default returns, empty outputs) are forbidden in core correctness paths; continue-but-observable is allowed only for explicitly salvage-mode training subpaths consuming model-generated outputs,
- remove redundant try/except wrappers that do not add actionable context (“strip over-engineering”).

## Findings (initial, high-signal patterns)

### P0 (silent swallow / semantics-changing fallback in core paths)

- `src/trainers/rollout_matching/parsing.py:326` — `except Exception: pass` used as control flow around `parse_coordjson(...)`.
  - Risk: suppresses all parse failures (including unexpected internal exceptions) and can misclassify a closed/invalid prefix as append-ready.
  - Recommendation: **Fix** — catch only the explicit parse/validation exception(s) expected for an *incomplete* prefix (or replace with an explicit “is_closed_container” predicate). Unexpected exceptions must propagate.

- `src/trainers/rollout_matching/matching.py:101` — `_mask_iou_norm1000` catches `Exception` and returns `0.0`.
  - Risk: pycocotools failures silently degrade matching quality and training supervision.
  - Recommendation: **Fix** — validate inputs before calling pycocotools; catch only known “invalid polygon” exceptions if needed and increment a counter; otherwise fail fast.

### P1 (silent swallowing in eval/infer or non-core helpers; needs narrowing/observability)

- `src/eval/detection.py:108` — `_segm_iou` catches `Exception` and returns `0.0`.
  - Risk: hides geometry conversion/pycocotools errors and silently biases metrics downward.
  - Recommendation: **Narrow/Fix** — validate segmentation inputs; catch only expected invalid-geometry exceptions and increment counters; unexpected exceptions should propagate.

- `src/common/prediction_parsing.py:96-97` — `load_prediction_dict` catches `Exception` and continues while parsing.
  - Note: this is salvage-oriented and upstream records `empty_pred`, but `except Exception` is too broad.
  - Recommendation: **Narrow** — catch `json.JSONDecodeError` + known transpiler validation exceptions; unexpected exceptions must propagate.

- `src/datasets/preprocessors/augmentation.py:95-97` — optional import guarded by `except Exception`, falling back to `Compose=None`.
  - Risk: hides non-import failures (e.g., runtime errors inside dependency import).
  - Recommendation: **Narrow** — catch only `ImportError` / `ModuleNotFoundError` and raise actionable guidance when augmentation is enabled.

- `src/datasets/preprocessors/augmentation.py:293-299` — `_coerce_value` catches `Exception` and returns the current value.
  - Risk: silent semantics change (config override is ignored) and hard-to-debug experiment drift.
  - Recommendation: **Fix** — replace with explicit type checks and catch only `(TypeError, ValueError)`; surface an error when a configured override cannot be coerced.

- `src/datasets/preprocessors/augmentation.py:270-272` — curriculum override of `bypass_prob` silently ignores invalid values via `except (TypeError, ValueError): pass`.
  - Risk: silent config drift.
  - Recommendation: **Fix** — at minimum log a warning (include the raw value); consider failing fast when augmentation curriculum is enabled.

- `src/datasets/preprocessors/resize.py:352-355` — fallback path computation uses `except Exception` to substitute a guessed `images/<filename>` path without diagnostics.
  - Risk: silently produces incorrect relative paths and can cascade into missing-file errors later.
  - Recommendation: **Fix** — catch `ValueError`/`RuntimeError` specifically, and add at least a warning counter/log when the “last resort” fallback is taken.

### P2 (over-engineering / best-effort I/O without diagnostics)

- `src/utils/logger.py:260-279` — file logging is disabled by returning `None` on blanket `Exception` without diagnostics.
  - Recommendation: **Narrow/Fix** — catch `OSError`/`PermissionError`, emit a warning once with path + exception type, and keep best-effort behavior I/O-only.

- `src/trainers/metrics/mixins.py:36-37` (and similar blocks throughout the file) — multiple `except Exception: raise` wrappers.
  - Risk: pure noise / over-engineering (no behavior change, harder to audit).
  - Recommendation: **Remove** — delete redundant wrappers; keep only handlers that add context or enforce a specific contract.

- `src/trainers/stage2_ab/scheduler.py:61-88` — repeated `try/except Exception` around reading integer args (per_device/world_size/gas).
  - Recommendation: **Narrow/Remove** — replace with explicit `getattr(..., default)` + `(TypeError, ValueError)` coercion; avoid blanket `Exception`.

- `src/trainers/stage2_ab/scheduler.py:142-145` — `_stage2_b_ratio_realized` catches `Exception` and returns `0.0` even though the loop is deterministic over a deque of ints.
  - Recommendation: **Remove** — delete the `try/except` and let unexpected exceptions surface (this code should not throw in normal operation).

- `src/coord_tokens/offset_adapter.py:215-216` and `src/optim/coord_offset_optimizer.py:15` — optional dependency imports guarded by `except Exception`.
  - Recommendation: **Narrow** — use `except ImportError`/`ModuleNotFoundError` (fail fast on other runtime errors).

- `src/datasets/preprocessors/sequential.py:30-31` and `src/datasets/preprocessors/sequential.py:44-45` — `except Exception: raise` wrappers.
  - Recommendation: **Remove** — no behavior change; rely on natural exception propagation.

- `src/datasets/preprocessors/resize.py:390-391` and `src/datasets/preprocessors/augmentation.py:251-252` — `except Exception: raise` wrappers.
  - Recommendation: **Remove** — no added context; keep code simpler to audit.

## Inventory status

This file is intentionally written in a “grow by evidence” style:
- The audit will be expanded to include **all** `try/except` blocks under `src/`, classified as **Keep / Narrow / Remove / Fix** with P0/P1/P2 severity where applicable.
- Directory-level audits are being collected (trainers, infer/eval, datasets/common, and remaining `src/` utilities/config/metrics/callbacks).

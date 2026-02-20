# silent-failure-policy Specification (delta: strict fail-fast; strip over-engineering)

## Purpose
Define a strict-by-default exception-handling policy so that core training/inference/evaluation behavior is reproducible and failures are observable, while allowing a narrow set of best-effort I/O sinks that do not affect correctness.

This delta expands the base `silent-failure-policy` definition of silent swallowing; all other base requirements remain unchanged unless explicitly modified below.

## Requirements

### Requirement: Core execution paths do not silently swallow unexpected exceptions
The system SHALL NOT suppress unexpected exceptions in core execution paths (dataset encoding, trainer steps, inference pipeline stages, evaluation metric computation).

Code MUST either:
- allow the exception to propagate (fail fast), OR
- catch only explicitly-enumerated exception types and either re-raise with added context, or (ONLY in explicitly salvage-mode model-output consumers) handle them as explicitly-defined model-output invalidity with structured errors + counters.

Enumeration MUST be expressed directly in code (e.g., `except (ExpectedModelOutputError, ...)`) and MUST NOT be driven by external allowlists/registries.

The following are forbidden in core execution paths:
- `except Exception: pass`
- `except: pass` (bare except)
- `except BaseException: pass`
- `except Exception: continue` (or equivalent suppression via `return <default>` / `break`)
- catching `Exception` / bare `except:` and returning semantics-changing “safe” defaults (e.g., `0.0`, empty dicts/lists, placeholder artifacts) without structured error recording and counters.

#### Scenario: Dataset encoding error is surfaced in training
- **WHEN** a dataset raises an exception while encoding a sample for training
- **THEN** the run fails fast with a clear error message
- **AND** the exception is not discarded by a blanket catch-all
- **AND** training does not continue with a silently-skipped sample.

### Requirement: Strip over-engineering in exception handling
Where a `try/except` block only re-raises the same exception without adding actionable context, it MUST be removed.

Where context is needed, it MUST be added once at a meaningful boundary using `raise ... from e`, not via repeated wrapper layers.

### Requirement: Operator-controlled input violations MUST fail fast (no skip-and-continue)
Operator-controlled inputs that can be validated in advance MUST be treated as strict contracts. Any sample-scoped validation/parse/contract failure MUST terminate the run with a non-zero exit code (fail fast); the system MUST NOT skip the sample and continue.

This includes:
- training inputs (dataset encoding, cooked targets, GT), and
- inference/eval inputs (JSONL format/schema, image path resolvability/readability, required width/height, geometry well-formedness, wrong format, etc.).

Implementations MAY emit a structured error record for the failing sample to aid debugging, but they MUST still raise and terminate the run.

#### Scenario: Inference fails fast on invalid input that can be validated in advance
- **GIVEN** inference/eval processing
- **WHEN** an input sample fails validation (e.g., invalid JSON line, missing image, malformed geometry, wrong schema)
- **THEN** the run terminates with a non-zero exit code
- **AND** the failure is surfaced with actionable diagnostics (sample identifier and reason).

### Requirement: Continue-but-observable is allowed ONLY for explicitly model-output consumers
When the system intentionally continues past model-generated output invalidity, it MUST:
- record a structured per-sample error entry (e.g., `errors=[...]`) in the relevant artifact, AND
- increment a run-level counter/metric for that error class, AND
- avoid emitting “fake success” outputs (e.g., empty predictions) without an accompanying error record.

This carve-out applies ONLY to explicitly model-output consumer codepaths (i.e., code that consumes model-generated outputs and must tolerate invalid/truncated predictions), such as:
- inference prediction parsing/validation, and
- salvage-mode training subpaths like rollout parsing/matching.

It MUST NOT be used for operator-controlled inputs or unexpected internal exceptions; those MUST fail fast.

Explicitly model-output consumer subpaths MAY continue past invalid model outputs per-sample, but MUST be observable (structured errors + counters) and MUST NOT suppress unexpected internal exceptions.

### Requirement: Structured errors + run-level counters have a minimal, machine-readable contract
When the system records per-sample errors, each error entry MUST include at least:
- `code` (stable string identifier),
- `message` (short human-readable summary),
- `stage` (stable stage identifier, e.g., `infer.parse_pred`, `infer.validate_pred`, `train.rollout_parse`).

If an exception is available, implementations SHOULD also include:
- `exception_type` (string), and
- `exception_message` (string).

Run-level counters MUST be machine-readable and MUST be persisted in a run artifact when such an artifact exists (e.g., inference `summary.json`). Logs alone are not sufficient.

At minimum, counters MUST include:
- `errors_total` (int), and
- `errors_by_code` (map `code -> count`).

If execution is distributed across ranks/processes, persisted counters MUST be globally aggregated, or explicitly labeled as rank-local.

### Requirement: Best-effort handling is limited to non-correctness sinks
Best-effort exception handling is allowed ONLY for explicitly sink-scoped code that cannot affect correctness-affecting state (e.g., log tee mirroring or diagnostics/telemetry reporting).

Such handlers MUST:
- catch narrow, expected exception types when possible (e.g., `OSError`, `PermissionError` for file I/O) rather than blanket `Exception`,
- emit explicit diagnostics (at least once),
- never suppress exceptions outside the sink itself.

Allowed best-effort sinks (non-exhaustive):
- log mirror tee writes,
- optional remote telemetry/reporting that is not required for correctness.

Forbidden best-effort sinks (non-exhaustive):
- writing canonical prediction artifacts and summaries (e.g., inference `gt_vs_pred.jsonl`, `summary.json`, resolved config artifacts),
- writing evaluator metric artifacts,
- saving checkpoints,
- mutating model/trainer state or correctness-affecting in-memory results.

#### Scenario: Log tee I/O failure does not abort training
- **WHEN** the file logging tee fails to write to its mirror file
- **THEN** training continues without corrupting model state
- **AND** exceptions in non-I/O code paths are not suppressed.

### Requirement: Blanket suppression is forbidden by direct CI scanning
CoordExp SHALL NOT maintain exception-suppression registries/allowlists. Compliance MUST be enforced directly by CI scanning source files.

Enforcement tiers (normative):

- **Tier 0 (blocking)**: the CI check in `tests/test_silent_failure_policy.py` MUST fail on:
  - `except Exception: pass` under `src/`
  - `except: pass` (bare except) under `src/`
  - `except BaseException: pass` under `src/`

- **Tier 1 (staged, then promoted to blocking)**: the CI check MUST also detect and report (with file + line) other silent suppression patterns in core paths, including:
  - `except Exception: continue` (or equivalent suppression via `break`),
  - `except Exception: return <default>` semantics-changing fallbacks,
  - catch-all handlers that set placeholder artifacts (empty preds/metrics) without structured errors + counters.

Tier 1 enforcement MAY begin as non-blocking to avoid brittle false positives, but it MUST NOT remain “warn-only” indefinitely:
- the change implementation MUST eliminate Tier 1 occurrences in core paths, and
- once the codebase has zero known Tier 1 occurrences under `src/`, Tier 1 checks MUST be promoted to blocking.

### Requirement: Temporary mutable state is restored deterministically
When code temporarily mutates shared mutable state to perform encoding (for example, overwriting `template.system` for one sample), the system MUST restore the original value in a `finally` block.

Failure to restore MUST terminate the run with an explicit error to prevent state leakage across samples.

#### Scenario: Template system prompt does not leak across samples
- **WHEN** encoding overrides `template.system` for one sample
- **THEN** the original value is restored before encoding the next sample
- **AND** restoration failure stops the run with an explicit error.

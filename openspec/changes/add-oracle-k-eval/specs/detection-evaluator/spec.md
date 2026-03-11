# detection-evaluator Specification (Delta)

## ADDED Requirements

### Requirement: Oracle-K repeated-sampling FN analysis is supported as an additive workflow
Detection-evaluator SHALL support an additive Oracle-K workflow that measures whether baseline false negatives are recoverable under repeated stochastic decoding and how often they are recovered across `K` samples.

Normative behavior:
- The workflow consumes:
  - one baseline run entry,
  - one or more Oracle run entries,
  - and a YAML config under `configs/eval/`.
- Each run entry MUST include:
  - `label`,
  - `pred_jsonl`.
- Each run entry MAY additionally include:
  - `pred_token_trace_jsonl`,
  - `resolved_config_json`.
- All artifacts MUST correspond to the same evaluation subset and MUST agree on:
  - record order,
  - width and height,
  - GT object content.
- Oracle-K MUST reuse the current F1-ish matching semantics for:
  - IoU threshold handling,
  - semantic matching,
  - prediction-scope filtering.
- Oracle-K primary-threshold selection MUST match the current evaluator contract:
  - prefer `0.50` when present,
  - otherwise use the largest requested IoU threshold.
- A baseline FN object MUST be analyzed separately under:
  - location-only recovery,
  - semantic+location recovery.
- For each view, the workflow MUST report:
  - whether the object is recovered at least once,
  - how many Oracle runs recover it,
  - its empirical recovery fraction across `K`.
- The normative join key for a baseline GT object is:
  - `record_idx`,
  - `gt_idx`.
- `image_id` and `file_name` are diagnostic-only fields and MUST NOT be treated as the normative cross-run join key.
- Oracle-K SHALL remain additive:
  - existing single-artifact evaluation behavior and artifacts MUST remain unchanged.

#### Scenario: A baseline FN is recovered at least once under Oracle-K
- **GIVEN** a baseline artifact where GT object `g` is unmatched under the primary threshold
- **AND** at least one Oracle artifact contains a full match for `g`
- **WHEN** Oracle-K analysis runs
- **THEN** `g` is labeled `ever_recovered_full=true`
- **AND** the summary counts it toward recoverable false negatives for the full-match view.

#### Scenario: A baseline FN has partial recovery frequency under Oracle-K
- **GIVEN** a baseline artifact where GT object `g` is unmatched under the primary threshold
- **AND** exactly 3 of 10 Oracle artifacts contain a location match for `g`
- **WHEN** Oracle-K analysis runs
- **THEN** the output records `recover_count_loc=3`
- **AND** `recover_fraction_loc=0.3`.

#### Scenario: A baseline FN is systematic under Oracle-K for the full-match view
- **GIVEN** a baseline artifact where GT object `g` is unmatched under the primary threshold
- **AND** no Oracle artifact contains a full match for `g`
- **WHEN** Oracle-K analysis runs
- **THEN** `g` is labeled `systematic_full=true`
- **AND** the summary counts it toward systematic false negatives for the full-match view.

#### Scenario: Oracle-K rejects misaligned artifacts
- **GIVEN** baseline and Oracle artifacts generated from different subsets or record orders
- **WHEN** Oracle-K analysis validates the artifacts
- **THEN** it fails fast with actionable diagnostics
- **AND** it does not emit partial Oracle-K metrics.

### Requirement: Oracle-K outputs are audit-friendly, GT-centric, and object-paired
The Oracle-K workflow SHALL write explicit artifacts for FN auditing and repeated-sampling recovery analysis.

Normative behavior:
- The workflow SHALL write:
  - `summary.json`,
  - `per_image.json`,
  - `fn_objects.jsonl`.
- `summary.json` MUST report, for each configured IoU threshold:
  - baseline TP / FN counts,
  - Oracle-K TP / FN counts,
  - baseline and Oracle-K recall-style metrics for:
    - location-only,
    - semantic+location,
  - `oracle_run_count`,
  - recoverable and systematic FN counts at the primary threshold for both views.
- `per_image.json` MUST provide per-image baseline FN totals and recoverable/systematic breakdowns for both views.
- `fn_objects.jsonl` MUST contain one row per baseline FN object with:
  - normative join keys (`record_idx`, `gt_idx`),
  - diagnostic identity (`file_name` when available),
  - GT description and geometry,
  - location-only and semantic+location recovery aggregates,
  - the Oracle run labels that recovered the object, when any,
  - per-run object-level pairing fields, including when present:
    - `matched_pred_idx`,
    - matched desc / bbox,
    - IoU,
    - semantic similarity / pass-fail.

#### Scenario: Oracle-K writes per-FN audit rows
- **GIVEN** an Oracle-K run with at least one baseline FN object
- **WHEN** the run completes
- **THEN** `fn_objects.jsonl` exists
- **AND** each row identifies one baseline FN object and its recovery status.

### Requirement: Oracle-K is YAML-first and allows repeated-sampling orchestration
The first Oracle-K contract SHALL be YAML-first and SHALL allow repeated-sampling orchestration without changing the existing standard inference-pipeline contract.

Normative behavior:
- The workflow SHALL be runnable from a dedicated script under `scripts/` with `--config`.
- A template config SHALL live under `configs/eval/`.
- The Oracle-K workflow MAY:
  - consume already-generated prediction artifacts,
  - or materialize repeated inference runs through a thin Oracle-K orchestrator.
- If repeated inference is materialized by the workflow, each generated Oracle run MUST still resolve to an explicit labeled run entry with persisted artifacts before aggregation begins.
- V1 SHALL NOT redefine or extend the existing `scripts/evaluate_detection.py` output schema.
- V1 SHOULD preserve `pred_token_trace_jsonl` provenance when available, but exact token-span-to-object alignment is not required.

#### Scenario: Oracle-K runs from a dedicated YAML config
- **GIVEN** a YAML config listing one baseline artifact and one or more Oracle artifacts
- **WHEN** the user runs the Oracle-K evaluator with `--config`
- **THEN** the workflow executes without requiring changes to the standard infer/eval pipeline artifacts.

#### Scenario: Oracle-K materializes repeated inference runs
- **GIVEN** a YAML config that defines one baseline run and multiple Oracle run specs
- **WHEN** the Oracle-K workflow materializes those runs
- **THEN** it persists labeled prediction artifacts for each run
- **AND** performs Oracle-K aggregation over those persisted artifacts.

### Requirement: Token-span-to-object alignment is not a v1 contract requirement
Oracle-K v1 SHALL prioritize object-level pairing over exact token-span-to-object alignment.

Normative behavior:
- V1 MUST provide object-level per-run pairing for baseline FN objects.
- V1 MAY preserve sample-level trace provenance for later inspection.
- V1 MUST NOT require exact token-span boundaries for matched predicted objects in order to be considered complete.

#### Scenario: Object-level pairing is sufficient for a valid v1 run
- **GIVEN** an Oracle-K run with per-run object-level pairing and no exact token-span alignment
- **WHEN** the workflow completes
- **THEN** the run satisfies the v1 Oracle-K contract.

# confidence-postop Specification

## Purpose
Define a CPU-only, offline post-operation that estimates one sortable confidence score per predicted object (initially `bbox_2d` only) so COCO-style AP/mAP ranking is meaningful and reproducible.

## Requirements

## ADDED Requirements

### Requirement: Inputs and join keys
The confidence post-operation SHALL consume:
- a unified inference artifact JSONL (canonical: `gt_vs_pred.jsonl`), and
- a token-trace sidecar JSONL (canonical: `pred_token_trace.jsonl`) containing per-token log-probabilities for each emitted sample.

The inference artifact records SHALL include `raw_output_json` (object or null) containing a strict JSON payload of the model output with an `objects` array, as produced by the shared salvage parser. The post-op uses this payload to reconstruct coord-token sequences for span matching.

The token-trace sidecar SHALL contain, per record:
- `line_idx` (int): 0-based JSONL line index in `gt_vs_pred.jsonl`,
- `generated_token_text` (list[string]): **all model-generated completion tokens** in order (excluding prompt/input tokens; no filtering/subsetting),
- `token_logprobs` (list[number]): **all** per-token natural-log probabilities (ln) aligned 1:1 with `generated_token_text` (no filtering/subsetting).

Coord-token contract:
- Coord special tokens MUST appear in `generated_token_text` as single elements of the exact form `<|coord_k|>` (not split into raw string fragments).

The post-op SHALL treat any non-finite `token_logprobs` values (`NaN`, `+/-inf`) as confidence computation failures for the affected object(s).

Join semantics:
- A trace record’s `line_idx` SHALL match the JSONL line index in `gt_vs_pred.jsonl`.
- If a trace record is missing for a sample, confidence computation SHALL NOT crash; all objects in that sample SHALL receive `confidence=null`, `score=null`, `kept=false` with an explicit `failure_reason`.
- If a trace record exists but `len(generated_token_text) != len(token_logprobs)`, confidence computation SHALL NOT crash; all objects in that sample SHALL receive `confidence=null`, `score=null`, `kept=false` with `failure_reason="trace_len_mismatch"`.

#### Scenario: Trace joins by `line_idx`
- **WHEN** a token-trace record has `line_idx = i`
- **THEN** the confidence post-op joins it to the i-th JSONL record in `gt_vs_pred.jsonl` deterministically.

#### Scenario: Missing raw payload yields `missing_coord_bins`
- **GIVEN** a `gt_vs_pred.jsonl` record with a `bbox_2d` prediction in `pred`
- **WHEN** the record has `raw_output_json=null` (or lacks a recoverable bbox coord-bin sequence)
- **THEN** the post-op emits `confidence=null`, `score=null`, `kept=false`, and `failure_reason=\"missing_coord_bins\"` for that object.

### Requirement: Confidence output schema is explicit and per-object
The confidence post-operation SHALL emit `pred_confidence.jsonl` with one record per sample:
- `line_idx` (int),
- `image` (string; copied from the inference artifact),
- `objects` (list[object]) where each object entry includes:
  - `object_idx` (int): index into the input record’s `pred` array (i.e., the emitted prediction list in `gt_vs_pred.jsonl` for that sample),
  - `type` (string),
  - `desc` (string),
  - `points` (list[number]),
  - `confidence` (number or null),
  - `score` (number or null; when confidence is non-null, `score` MUST equal `confidence`; when confidence is null, `score` MUST be null),
  - `kept` (bool; true iff this object will be included in the scored artifact’s `pred` array),
  - `confidence_details` (object) including:
    - `method` (string),
    - `coord_token_count` (int),
    - `matched_token_indices` (list[int]),
    - `ambiguous_matches` (int),
    - `failure_reason` (string or null).

The post-op SHALL emit `objects` in increasing `object_idx` order, and SHALL NOT reorder objects.

The `kept` flag SHALL be consistent with the scored artifact:
- If `kept=true`, then `confidence` MUST be non-null and `score` MUST be a finite number satisfying `0.0 < score <= 1.0`.
- If `kept=false`, then `confidence` MUST be null and `score` MUST be null.

Unsupported geometry types (including `poly`) SHALL produce `confidence=null`, `score=null`, `kept=false` and a deterministic `failure_reason` (e.g., `unsupported_geometry_type`).

#### Scenario: Unsupported geometry yields null confidence
- **WHEN** a predicted object has `type != "bbox_2d"`
- **THEN** the output object has `confidence=null`, `score=null`, `kept=false`, and `confidence_details.failure_reason=\"unsupported_geometry_type\"`.

### Requirement: Default bbox confidence definition
For each `bbox_2d` prediction where exactly 4 bbox coord tokens can be aligned to token log-probabilities:
- Let aligned coord-token logprobs be `lp1..lp4` in bbox coordinate order `(x1, y1, x2, y2)`.
- The reducer SHALL be `mean_logprob = (lp1 + lp2 + lp3 + lp4) / 4`.
- The mapping SHALL be `score = exp(mean_logprob)` (a value in `(0, 1]`).

Confidence computation SHALL preserve bbox coordinate order and SHALL NOT reorder/drop coordinates inside a matched bbox slot.

If the computed `score` is not finite or does not satisfy `0.0 < score <= 1.0`, confidence computation SHALL be treated as a failure for that object and the object MUST NOT be kept in the scored artifact.

#### Scenario: Bbox confidence computed from 4 coord tokens
- **WHEN** a predicted bbox is aligned to 4 coord tokens with logprobs `[-0.1, -0.2, -0.3, -0.4]`
- **THEN** the emitted `confidence` equals `exp(mean([-0.1,-0.2,-0.3,-0.4]))` and `score` equals `confidence`.

### Requirement: Span resolution is deterministic under repeated coord patterns
Span resolution (object → token indices) SHALL be deterministic:
- Reconstruct the expected coord-token sequence for the object and search it as an exact subsequence in `generated_token_text`.
- Assign matches left-to-right in predicted-object order.
- If multiple candidate matches exist for an object, choose the earliest unused match and record ambiguity via `confidence_details.ambiguous_matches` (zero when unambiguous).

If fewer than 4 coord tokens can be aligned, the object SHALL receive `confidence=null`, `score=null`, `kept=false` with `failure_reason=\"missing_span\"` (the system MUST NOT fabricate partial confidence).

#### Scenario: Repeated coord-token sequences choose earliest unused match
- **WHEN** the same coord-token sequence appears multiple times in a token trace
- **THEN** span resolution chooses the earliest unused match, preserving object order deterministically.

### Requirement: Object alignment is validated against emitted predictions (no silent drift)
The post-op MUST NOT silently assign confidence scores to the wrong objects.

If the post-op reconstructs objects from `raw_output_json` (e.g., by reusing the shared parsing/standardization logic to reproduce the emitted `pred` list), it MUST validate that the reconstructed standardized object list matches the input record’s `pred` list (same length and per-index equality on at least `type`, `points`, and `desc` after normalization).

Normalization for alignment checks:
- `desc_norm = desc.strip()` (whitespace trim only; no lowercasing or other rewriting).

If validation fails for a sample, the post-op SHALL emit `confidence=null`, `score=null`, `kept=false`, and `failure_reason=\"pred_alignment_mismatch\"` for every object in that sample (and MUST NOT keep any object from that sample in the scored artifact).

#### Scenario: Pred alignment mismatch yields `pred_alignment_mismatch`
- **GIVEN** a `gt_vs_pred.jsonl` record where the post-op cannot deterministically align raw objects to `pred[*]`
- **WHEN** alignment validation fails
- **THEN** all objects in that sample get `confidence=null`, `score=null`, `kept=false` with `failure_reason=\"pred_alignment_mismatch\"` and no ambiguous partial merge occurs.

### Requirement: Coord-token sequence reconstruction uses raw output bins (no pixel inversion)
To reconstruct the expected bbox coord-token sequence (`<|coord_k|>`), the post-op MUST use the model-output bin values from the inference artifact’s raw payload (e.g., `raw_output_json`), not the pixel-space `pred[*].points` values.

The post-op MUST NOT attempt to invert pixel points back into norm1000 bins, as this is lossy under clamp/round/denorm.

If the raw payload is missing or does not contain a recoverable bbox coord-token sequence, the object SHALL receive `confidence=null`, `score=null`, `kept=false`, and `failure_reason=\"missing_coord_bins\"`.

#### Scenario: Pixel-space points are not inverted for matching
- **GIVEN** a `gt_vs_pred.jsonl` record whose `pred[*].points` are pixel-space ints
- **WHEN** confidence post-op reconstructs coord-token sequences for matching
- **THEN** it uses raw model-output bin values from the raw payload, and does not infer bins from pixel points.

### Requirement: Failure reasons are enumerated and stable
When `confidence` is null, `confidence_details.failure_reason` MUST be one of the following stable snake_case strings:
- `missing_trace`: no trace record available for the sample.
- `trace_len_mismatch`: `generated_token_text` and `token_logprobs` lengths differ.
- `unsupported_geometry_type`: object is not `bbox_2d`.
- `missing_coord_bins`: raw payload lacks a recoverable 4-token bbox coord sequence.
- `missing_span`: coord-token sequence cannot be matched to the token trace.
- `nonfinite_logprob`: one or more matched token logprobs are non-finite.
- `pred_alignment_mismatch`: raw-object ↔ emitted-pred alignment could not be validated deterministically.
- `object_idx_oob`: confidence object_idx does not index into the input `pred` list (implementation bug; MUST be observable).

#### Scenario: Missing trace yields `missing_trace`
- **GIVEN** a `gt_vs_pred.jsonl` record with at least one prediction
- **WHEN** the corresponding token-trace record for that `line_idx` is missing
- **THEN** each object has `confidence=null`, `score=null`, `kept=false`, and `failure_reason=\"missing_trace\"`.

### Requirement: Scored artifact merge is auditable and non-destructive
The post-op SHALL write a derived scored artifact `gt_vs_pred_scored.jsonl` by:
- copying each input record from `gt_vs_pred.jsonl` verbatim except for `pred` (and adding score-provenance metadata keys),
- building a new `pred` array by iterating over the input record’s `pred` objects in order and:
  - keeping an object only if it has a computed, finite confidence-derived `score`, and
  - dropping the object otherwise (including `confidence=null` cases).

The scored artifact MUST include per-record metadata keys:
- `pred_score_source` (string, non-empty; set to `confidence_postop` for this post-op), and
- `pred_score_version` (int; initial version `1`).

The base `gt_vs_pred.jsonl` artifact SHALL NOT be modified in-place.

#### Scenario: Scored JSONL merges confidence into `pred[*].score`
- **WHEN** confidence post-op computes `score=0.8` for `object_idx=0` on `line_idx=i`
- **THEN** `gt_vs_pred_scored.jsonl` record i contains that object in `pred` with `score=0.8` and other fields unchanged.

#### Scenario: Objects with missing confidence are dropped from scored predictions
- **GIVEN** a `gt_vs_pred.jsonl` record with two predictions where one object has `confidence=null`
- **WHEN** the scored artifact is written
- **THEN** the `pred` array excludes the object with `confidence=null` and preserves the input order among retained objects.

### Requirement: Post-op emits a deterministic drop/keep summary
The post-op MUST emit a run-level summary JSON (`confidence_postop_summary.json`) that makes dropping auditable without inspecting `pred_confidence.jsonl` directly.

The summary MUST include at least:
- `total_samples` (int),
- `total_pred_objects` (int),
- `kept_pred_objects` (int),
- `dropped_pred_objects` (int),
- `kept_fraction` (number in `[0.0, 1.0]`; defined as `kept_pred_objects / total_pred_objects` when `total_pred_objects>0`, else `1.0`),
- `dropped_by_reason` (object mapping `failure_reason` → int count; only reasons from this spec are permitted as keys),
- `pred_score_source` (string; MUST be `confidence_postop`),
- `pred_score_version` (int; MUST be `1`).

The summary MUST be deterministic for the same inputs.

#### Scenario: Missing trace is observable in the summary
- **GIVEN** a run where at least one sample lacks a trace record (`missing_trace`)
- **WHEN** the post-op finishes
- **THEN** `confidence_postop_summary.json` reports a non-zero `dropped_by_reason.missing_trace` and `kept_fraction < 1.0`.

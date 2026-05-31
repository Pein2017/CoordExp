## ADDED Requirements

### Requirement: Evaluator distinguishes DetectionEvalRecord from scored eval records

The clean-break evaluator SHALL use explicit raw and scored eval-record concepts
while keeping current artifact filenames stable.

Normative behavior:

- raw GT/pred comparison rows are `DetectionEvalRecord` values;
- scored comparison rows with IoU, match, score, or metric annotations are
  `ScoredDetectionEvalRecord` values;
- `gt_vs_pred.jsonl` remains the raw artifact filename for now;
- `gt_vs_pred_scored.jsonl` remains the scored artifact filename for now;
- re-scoring or metric annotation MUST produce scored records and MUST NOT
  mutate raw records in place;
- any future artifact filename rename MUST be a separate artifact-contract
  decision.

#### Scenario: Raw and scored eval records remain conceptually separate

- **GIVEN** a raw detection eval artifact has been materialized
- **WHEN** scoring is applied for score-aware evaluation
- **THEN** the result is represented as `ScoredDetectionEvalRecord`
- **AND** the raw `DetectionEvalRecord` interpretation remains unchanged.

## ADDED Requirements

### Requirement: Inference decode produces DecodedDetectionResult for detection workflows

Clean-break detection inference SHALL represent parsed generation output as
`DecodedDetectionResult` before eval artifact materialization.

Normative behavior:

- `DecodedDetectionResult` MUST contain parsed predictions plus invalid/drop
  metadata;
- strict parser results used for official eval MUST preserve metric-bearing
  status and salvage exclusion;
- inference decode MUST NOT expose rendered text, raw backend output, or eval
  rows as the semantic prediction object for downstream detection logic;
- raw artifact filenames `gt_vs_pred.jsonl` and scored artifact filenames
  `gt_vs_pred_scored.jsonl` remain stable unless a separate artifact-contract
  change renames them.

#### Scenario: Eval artifact materialization consumes decoded detection results

- **GIVEN** model generation output for a detection image
- **WHEN** inference parsing succeeds for official eval
- **THEN** the parsed output is represented as `DecodedDetectionResult`
- **AND** eval artifact rows are materialized from that decoded result rather
  than from raw generated text.

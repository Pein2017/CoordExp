## ADDED Requirements

### Requirement: Post-hoc mAP validates compact row provenance without changing metric semantics
Post-hoc mAP SHALL require compact artifact provenance to include detection
template id and object field order when evaluating post-change compact outputs.

mAP computation SHALL continue to score normalized object arrays and pixel-space
geometry.  It MUST NOT reparse raw compact text or infer bbox-first versus
desc-first from raw generated strings in the standard metric path.

#### Scenario: Equivalent normalized predictions have equal mAP
- **GIVEN** two compact artifact families with different object field orders
- **AND** their normalized `pred` object arrays and ground truth are equivalent
- **WHEN** post-hoc mAP runs
- **THEN** metric computation receives equivalent normalized predictions
- **AND** the mAP values are equal.

#### Scenario: Missing field-order metadata fails preflight
- **GIVEN** a post-change compact artifact family without object field order
  metadata
- **WHEN** post-hoc mAP is requested
- **THEN** evaluation fails before scoring
- **AND** the error instructs the operator to regenerate or repair artifact
  metadata.
